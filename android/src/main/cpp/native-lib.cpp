#include <android/log.h>
#include <jni.h>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <chrono>

// KataGo Headers
#include "katago/cpp/core/global.h"
#include "katago/cpp/core/config_parser.h"
#include "katago/cpp/game/board.h"
#include "katago/cpp/game/boardhistory.h"
#include "katago/cpp/game/rules.h"
#include "katago/cpp/neuralnet/nneval.h"
#include "katago/cpp/neuralnet/nninputs.h"
#include "katago/cpp/search/search.h"
#include "katago/cpp/search/searchparams.h"
#include "katago/cpp/external/nlohmann_json/json.hpp"

#define TAG "KataGoNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using json = nlohmann::json;

// ============================================================================
// Global State (no threads, no pipes)
// ============================================================================

static Logger* g_logger = nullptr;
static NNEvaluator* g_nnEval = nullptr;
static SearchParams* g_searchParams = nullptr;
static Rules g_rules;
static std::string g_modelName = "kata1-b6c96";
std::string g_onnxModelPath;  // Global for onnxbackend.cpp
static bool g_globalsInitialized = false;  // Track one-time global init

// ============================================================================
// Helper Functions
// ============================================================================

// Parse GTP coordinate (e.g., "Q16") to Loc
static Loc parseGTPLoc(const std::string& s, int boardXSize, int boardYSize) {
  if (s == "pass" || s == "PASS") {
    return Board::PASS_LOC;
  }

  if (s.length() < 2) return Board::NULL_LOC;

  char col = s[0];
  int row;
  std::istringstream rowStream(s.substr(1));
  rowStream >> row;

  // GTP format: A-T (skip I), 1-19
  int x, y;
  if (col >= 'A' && col <= 'Z') {
    x = col - 'A';
    if (col >= 'I') x--;  // Skip 'I'
  } else if (col >= 'a' && col <= 'z') {
    x = col - 'a';
    if (col >= 'i') x--;
  } else {
    return Board::NULL_LOC;
  }

  y = boardYSize - row;  // GTP row 1 = bottom

  if (x < 0 || x >= boardXSize || y < 0 || y >= boardYSize) {
    return Board::NULL_LOC;
  }

  return Location::getLoc(x, y, boardXSize);
}

// Convert Loc to GTP coordinate
static std::string locToGTP(Loc loc, int boardXSize, int boardYSize) {
  if (loc == Board::PASS_LOC) {
    return "pass";
  }
  if (loc == Board::NULL_LOC) {
    return "null";
  }

  int x = Location::getX(loc, boardXSize);
  int y = Location::getY(loc, boardXSize);

  char col = 'A' + x;
  if (col >= 'I') col++;  // Skip 'I'

  int row = boardYSize - y;

  return std::string(1, col) + std::to_string(row);
}

// ============================================================================
// JNI Methods
// ============================================================================

extern "C" JNIEXPORT jboolean JNICALL
Java_com_justmaker_katago_1onnx_1mobile_KataGoEngine_initializeNative(
    JNIEnv* env,
    jobject thiz,
    jstring configPath,
    jstring modelBinPath,
    jstring modelOnnxPath,
    jint boardSize) {

  const char* cfgStr = env->GetStringUTFChars(configPath, nullptr);
  const char* binStr = env->GetStringUTFChars(modelBinPath, nullptr);
  const char* onnxStr = env->GetStringUTFChars(modelOnnxPath, nullptr);

  std::string configFile(cfgStr);
  std::string modelBinFile(binStr);
  std::string modelOnnxFile(onnxStr);

  env->ReleaseStringUTFChars(configPath, cfgStr);
  env->ReleaseStringUTFChars(modelBinPath, binStr);
  env->ReleaseStringUTFChars(modelOnnxPath, onnxStr);

  LOGI("=== Initializing KataGo (ONNX Backend, Single-threaded) ===");
  LOGI("Config: %s", configFile.c_str());
  LOGI("Model (bin.gz): %s", modelBinFile.c_str());
  LOGI("Model (onnx): %s", modelOnnxFile.c_str());
  LOGI("Board size: %dx%d", boardSize, boardSize);

  // Set global ONNX model path for onnxbackend.cpp
  g_onnxModelPath = modelOnnxFile;

  try {
    // 1. Initialize logger (reuse if exists)
    if (g_logger == nullptr) {
      g_logger = new Logger(nullptr, false, false, false, false);
    }

    // 2. One-time global initialization (must skip on reinit to avoid crash)
    if (!g_globalsInitialized) {
      Board::initHash();
      LOGI("✓ Board zobrist hash initialized");
      ScoreValue::initTables();
      LOGI("✓ ScoreValue tables initialized");
      NeuralNet::globalInitialize();
      LOGI("✓ NeuralNet backend initialized");
      g_globalsInitialized = true;
    } else {
      LOGI("Reinit: reusing global state (Board hash, ScoreValue, NeuralNet)");
    }

    // 3. Parse config
    ConfigParser cfg(configFile);

    // Force single-threaded configuration
    int numSearchThreads = 1;
    int maxVisits = cfg.getInt("maxVisits", 1, 1000000000);
    int nnCacheSizePowerOfTwo = cfg.getInt("nnCacheSizePowerOfTwo", 0, 48);

    LOGI("maxVisits: %d", maxVisits);
    LOGI("numSearchThreads: %d (forced single-threaded)", numSearchThreads);

    // 4. Load model (LoadedModel will load both .bin.gz and .onnx)
    // IMPORTANT: For ONNX backend, loadModelFile expects .bin.gz path
    // and automatically finds the .onnx file
    LoadedModel* loadedModel = NeuralNet::loadModelFile(modelBinFile, "");

    const ModelDesc& modelDesc = NeuralNet::getModelDesc(loadedModel);
    LOGI("Model loaded: %s, version %d", modelDesc.name.c_str(), modelDesc.modelVersion);

    // 5. Set board size for NN evaluation (must match ONNX model dimensions)
    int nnXLen = boardSize > 0 ? boardSize : 19;
    int nnYLen = boardSize > 0 ? boardSize : 19;
    LOGI("NN board size: %dx%d", nnXLen, nnYLen);

    // 6. Create NNEvaluator (single-threaded mode)
    std::vector<int> gpuIdxs = {-1};  // Default GPU
    g_nnEval = new NNEvaluator(
      g_modelName,
      modelBinFile,
      "",  // expectedSha256
      g_logger,
      1,   // maxBatchSize = 1 (single-threaded)
      nnXLen,
      nnYLen,
      false, // requireExactNNLen
      true,  // inputsUseNHWC
      nnCacheSizePowerOfTwo,
      17,    // mutexPoolSize
      false, // debugSkipNeuralNet
      "",    // openCLTunerFile
      "",    // homeDataDirOverride
      false, // openCLReTunePerBoardSize
      enabled_t::Auto, // useFP16
      enabled_t::Auto, // useNHWC
      numSearchThreads,
      gpuIdxs,
      "androidSeed", // randSeed
      false, // doRandomize
      0      // defaultSymmetry
    );

    // CRITICAL: Enable single-threaded mode to avoid pthread
    g_nnEval->setSingleThreadedMode(true);
    LOGI("✓ Single-threaded mode enabled");

    // DO NOT call spawnServerThreads() - we use single-threaded mode

    // 7. Create SearchParams
    g_searchParams = new SearchParams();
    g_searchParams->numThreads = numSearchThreads;
    g_searchParams->maxVisits = maxVisits;
    g_searchParams->maxPlayouts = maxVisits;
    g_searchParams->maxTime = 1e30;  // No time limit
    g_searchParams->lagBuffer = 0.0;
    g_searchParams->searchFactorAfterOnePass = 1.0;
    g_searchParams->searchFactorAfterTwoPass = 1.0;

    // 8. Setup default rules (Chinese rules, 7.5 komi)
    g_rules = Rules::getTrompTaylorish();
    g_rules.komi = 7.5f;

    LOGI("✓ KataGo initialized successfully (no pthread created)");
    return JNI_TRUE;

  } catch (const StringError& e) {
    LOGE("Initialization failed: %s", e.what());
    return JNI_FALSE;
  } catch (const std::exception& e) {
    LOGE("Initialization exception: %s", e.what());
    return JNI_FALSE;
  }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_justmaker_katago_1onnx_1mobile_KataGoEngine_analyzePositionNative(
    JNIEnv* env,
    jobject thiz,
    jint boardXSize,
    jint boardYSize,
    jdouble komi,
    jint maxVisits,
    jobjectArray movesArray) {

  LOGI("=== analyzePositionNative ===");
  LOGI("Board: %dx%d, Komi: %.1f, MaxVisits: %d", boardXSize, boardYSize, komi, maxVisits);

  auto timeStart = std::chrono::steady_clock::now();
  auto timeMark = [&timeStart]() -> double {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(now - timeStart).count();
  };

  try {
    // 1. Parse moves array
    jsize numMoves = env->GetArrayLength(movesArray);
    LOGI("[%.3fs] Parsing %d moves...", timeMark(), numMoves);

    std::vector<std::pair<Player, Loc>> moves;
    for (jsize i = 0; i < numMoves; i++) {
      jobjectArray moveArray = (jobjectArray)env->GetObjectArrayElement(movesArray, i);
      jstring colorStr = (jstring)env->GetObjectArrayElement(moveArray, 0);
      jstring locStr = (jstring)env->GetObjectArrayElement(moveArray, 1);

      const char* colorChars = env->GetStringUTFChars(colorStr, nullptr);
      const char* locChars = env->GetStringUTFChars(locStr, nullptr);

      Player pla = (colorChars[0] == 'B' || colorChars[0] == 'b') ? P_BLACK : P_WHITE;
      Loc loc = parseGTPLoc(std::string(locChars), boardXSize, boardYSize);

      env->ReleaseStringUTFChars(colorStr, colorChars);
      env->ReleaseStringUTFChars(locStr, locChars);
      env->DeleteLocalRef(colorStr);
      env->DeleteLocalRef(locStr);
      env->DeleteLocalRef(moveArray);

      if (loc != Board::NULL_LOC) {
        moves.push_back({pla, loc});
      }
    }

    // 2. Build Board and BoardHistory
    LOGI("[%.3fs] Building board...", timeMark());
    Board board(boardXSize, boardYSize);
    Player nextPla = P_BLACK;
    BoardHistory history(board, nextPla, g_rules, 0);
    history.setKomi((float)komi);

    for (const auto& move : moves) {
      if (!history.isLegal(board, move.second, move.first)) {
        LOGE("Illegal move: %s %s",
             PlayerIO::playerToString(move.first).c_str(),
             locToGTP(move.second, boardXSize, boardYSize).c_str());
        continue;
      }
      history.makeBoardMoveAssumeLegal(board, move.second, move.first, nullptr);
      nextPla = getOpp(move.first);
    }

    LOGI("[%.3fs] Position set up, next player: %s", timeMark(),
         PlayerIO::playerToString(nextPla).c_str());

    // 3. Create Search (single-threaded)
    SearchParams searchParams = *g_searchParams;
    searchParams.maxVisits = maxVisits;
    searchParams.maxPlayouts = maxVisits;

    LOGI("[%.3fs] Creating Search object...", timeMark());
    Search* search = new Search(searchParams, g_nnEval, g_logger, "androidSearch");

    // 4. Set position
    LOGI("[%.3fs] Setting position...", timeMark());
    search->setPosition(nextPla, board, history);

    // 5. Run search (synchronous, single-threaded, no pthread)
    LOGI("[%.3fs] Starting search (%d visits)...", timeMark(), maxVisits);
    search->runWholeSearch(nextPla);
    LOGI("[%.3fs] Search completed", timeMark());

    // 6. Extract full analysis results using KataGo's built-in JSON export
    LOGI("[%.3fs] Extracting JSON...", timeMark());
    json result;
    bool suc = search->getAnalysisJson(
      P_BLACK,   // perspective: always report winrates from Black's perspective
      7,         // analysisPVLen
      false,     // preventEncore
      false,     // includePolicy
      false,     // includeOwnership
      false,     // includeOwnershipStdev
      false,     // includeMovesOwnership
      false,     // includeMovesOwnershipStdev
      false,     // includePVVisits
      false,     // includeNoResultValue
      result
    );

    if (!suc) {
      LOGE("getAnalysisJson failed, falling back to empty result");
      result["moveInfos"] = json::array();
      result["rootInfo"] = json::object();
    }

    result["id"] = "android_analysis";
    result["turnNumber"] = (int)history.moveHistory.size();

    // 7. Cleanup
    LOGI("[%.3fs] Deleting search...", timeMark());
    delete search;
    LOGI("[%.3fs] Search deleted", timeMark());

    // 8. Return JSON
    std::string jsonStr = result.dump();
    LOGI("[%.3fs] Analysis result: %zu bytes", timeMark(), jsonStr.length());

    return env->NewStringUTF(jsonStr.c_str());

  } catch (const StringError& e) {
    LOGE("Analysis failed: %s", e.what());
    return env->NewStringUTF("{\"error\": \"Analysis failed\"}");
  } catch (const std::exception& e) {
    LOGE("Analysis exception: %s", e.what());
    return env->NewStringUTF("{\"error\": \"Exception occurred\"}");
  }
}

extern "C" JNIEXPORT void JNICALL
Java_com_justmaker_katago_1onnx_1mobile_KataGoEngine_destroyNative(
    JNIEnv* env,
    jobject thiz) {

  LOGI("=== Destroying KataGo ===");

  if (g_nnEval != nullptr) {
    delete g_nnEval;
    g_nnEval = nullptr;
  }

  if (g_searchParams != nullptr) {
    delete g_searchParams;
    g_searchParams = nullptr;
  }

  // Keep logger and global state alive for reinit
  // NeuralNet::globalCleanup() is NOT called here to allow reinit

  LOGI("✓ KataGo destroyed (globals preserved for reinit)");
}

// JNI_OnLoad: Early initialization
JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved) {
  LOGI("JNI_OnLoad called - ONNX backend, single-threaded mode");
  return JNI_VERSION_1_6;
}
