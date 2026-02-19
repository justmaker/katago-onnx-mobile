//
//  KataGoOnnxBridge.mm
//  KataGoMobile
//
//  ONNX Runtime-based KataGo analysis for iOS
//  Ported from Android native-lib.cpp
//

#import "KataGoOnnxBridge.h"
#import <Foundation/Foundation.h>

// C++ Standard Library
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
#include "katago/cpp/search/timecontrols.h"
#include "katago/cpp/external/nlohmann_json/json.hpp"

using json = nlohmann::json;

// ============================================================================
// Global State (no threads, no pipes - same as Android)
// ============================================================================

static Logger* g_logger = nullptr;
static NNEvaluator* g_nnEval = nullptr;
static SearchParams* g_searchParams = nullptr;
static Rules g_rules;
static std::string g_modelName = "g170-b20c256x2";
std::string g_onnxModelPath;  // Global for onnxbackend.cpp
static bool g_globalsInitialized = false;
static bool g_engineInitialized = false;

// Progress tracking state (thread-safe reads via atomic/pointer)
static Search* g_activeSearch = nullptr;       // Points to Search during analysis
static int g_currentMaxVisits = 0;             // maxVisits for current analysis
static std::atomic<bool> g_shouldStop(false);  // Atomic stop flag

// Logging macros
#define LOG_INFO(fmt, ...) NSLog(@"[KataGoONNX] " fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) NSLog(@"[KataGoONNX ERROR] " fmt, ##__VA_ARGS__)

// ============================================================================
// Helper Functions (ported from Android)
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
// Implementation
// ============================================================================

@implementation KataGoOnnxBridge

+ (BOOL)initializeWithConfig:(NSString *)configPath
                    modelBin:(NSString *)modelBinPath
                   modelOnnx:(NSString *)modelOnnxPath
                   boardSize:(int)boardSize {

    @autoreleasepool {
        std::string configFile = [configPath UTF8String];
        std::string modelBinFile = [modelBinPath UTF8String];
        std::string modelOnnxFile = [modelOnnxPath UTF8String];

        LOG_INFO(@"=== Initializing KataGo (ONNX Backend, Single-threaded) ===");
        LOG_INFO(@"Config: %s", configFile.c_str());
        LOG_INFO(@"Model (bin.gz): %s", modelBinFile.c_str());
        LOG_INFO(@"Model (onnx): %s", modelOnnxFile.c_str());
        LOG_INFO(@"Board size: %dx%d", boardSize, boardSize);

        // Set global ONNX model path for onnxbackend.cpp
        g_onnxModelPath = modelOnnxFile;

        @try {
            // 1. Initialize logger (reuse if exists)
            if (g_logger == nullptr) {
                g_logger = new Logger(nullptr, false, false, false, false);
            }

            // 2. One-time global initialization
            if (!g_globalsInitialized) {
                Board::initHash();
                LOG_INFO(@"✓ Board zobrist hash initialized");

                ScoreValue::initTables();
                LOG_INFO(@"✓ ScoreValue tables initialized");

                NeuralNet::globalInitialize();
                LOG_INFO(@"✓ NeuralNet backend initialized");

                g_globalsInitialized = true;
            } else {
                LOG_INFO(@"Reinit: reusing global state");
            }

            // 3. Parse config
            ConfigParser cfg(configFile);

            // Force single-threaded configuration (CRITICAL for mobile)
            int numSearchThreads = 1;
            int maxVisits = cfg.getInt("maxVisits", 1, 1000000000);
            int nnCacheSizePowerOfTwo = cfg.getInt("nnCacheSizePowerOfTwo", 0, 48);

            LOG_INFO(@"maxVisits: %d", maxVisits);
            LOG_INFO(@"numSearchThreads: %d (forced single-threaded)", numSearchThreads);

            // 4. Load model
            // For ONNX backend, loadModelFile expects .bin.gz path
            // and automatically finds the .onnx file via g_onnxModelPath
            LoadedModel* loadedModel = NeuralNet::loadModelFile(modelBinFile, "");

            const ModelDesc& modelDesc = NeuralNet::getModelDesc(loadedModel);
            LOG_INFO(@"Model loaded: %s, version %d", modelDesc.name.c_str(), modelDesc.modelVersion);

            // 5. Set board size for NN evaluation
            int nnXLen = boardSize > 0 ? boardSize : 19;
            int nnYLen = boardSize > 0 ? boardSize : 19;
            LOG_INFO(@"NN board size: %dx%d", nnXLen, nnYLen);

            // 6. Create NNEvaluator (single-threaded mode)
            std::vector<int> gpuIdxs = {-1};  // CPU only

            if (g_nnEval != nullptr) {
                delete g_nnEval;
                g_nnEval = nullptr;
            }

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
                "iOSSeed", // randSeed
                false, // doRandomize
                0      // defaultSymmetry
            );

            // CRITICAL: Single-threaded mode - evaluate on calling thread directly
            g_nnEval->setSingleThreadedMode(true);
            LOG_INFO(@"✓ Single-threaded mode enabled");

            // DO NOT call spawnServerThreads() - we use single-threaded mode

            // 7. Create SearchParams
            if (g_searchParams != nullptr) {
                delete g_searchParams;
            }
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

            LOG_INFO(@"✓ KataGo initialized successfully (no pthread created)");
            g_engineInitialized = true;
            return YES;

        } @catch (NSException *exception) {
            LOG_ERROR(@"Initialization failed: %@", exception.reason);
            return NO;
        }
    }
}

+ (nullable NSString *)analyzePosition:(NSDictionary *)params {
    @autoreleasepool {
        if (!g_engineInitialized || g_nnEval == nullptr || g_searchParams == nullptr) {
            LOG_ERROR(@"Engine not initialized");
            return nil;
        }

        // Extract parameters
        int boardXSize = [[params objectForKey:@"boardXSize"] intValue];
        int boardYSize = [[params objectForKey:@"boardYSize"] intValue];
        double komi = [[params objectForKey:@"komi"] doubleValue];
        int maxVisits = [[params objectForKey:@"maxVisits"] intValue];
        NSArray *movesArray = [params objectForKey:@"moves"];

        LOG_INFO(@"=== analyzePosition ===");
        LOG_INFO(@"Board: %dx%d, Komi: %.1f, MaxVisits: %d", boardXSize, boardYSize, komi, maxVisits);

        auto timeStart = std::chrono::steady_clock::now();
        auto timeMark = [&timeStart]() -> double {
            auto now = std::chrono::steady_clock::now();
            return std::chrono::duration<double>(now - timeStart).count();
        };

        @try {
            // 1. Parse moves array
            LOG_INFO(@"[%.3fs] Parsing %lu moves...", timeMark(), (unsigned long)[movesArray count]);

            std::vector<std::pair<Player, Loc>> moves;
            for (NSArray *moveArray in movesArray) {
                if ([moveArray count] != 2) continue;

                NSString *colorStr = moveArray[0];
                NSString *locStr = moveArray[1];

                Player pla = ([colorStr characterAtIndex:0] == 'B' || [colorStr characterAtIndex:0] == 'b')
                             ? P_BLACK : P_WHITE;
                Loc loc = parseGTPLoc([locStr UTF8String], boardXSize, boardYSize);

                if (loc != Board::NULL_LOC) {
                    moves.push_back({pla, loc});
                }
            }

            // 2. Build Board and BoardHistory
            LOG_INFO(@"[%.3fs] Building board...", timeMark());
            Board board(boardXSize, boardYSize);
            Player nextPla = P_BLACK;
            BoardHistory history(board, nextPla, g_rules, 0);
            history.setKomi((float)komi);

            for (const auto& move : moves) {
                if (!history.isLegal(board, move.second, move.first)) {
                    LOG_ERROR(@"Illegal move: %s %s",
                             PlayerIO::playerToString(move.first).c_str(),
                             locToGTP(move.second, boardXSize, boardYSize).c_str());
                    continue;
                }
                history.makeBoardMoveAssumeLegal(board, move.second, move.first, nullptr);
                nextPla = getOpp(move.first);
            }

            LOG_INFO(@"[%.3fs] Position set up, next player: %s", timeMark(),
                     PlayerIO::playerToString(nextPla).c_str());

            // 3. Create Search (single-threaded)
            SearchParams searchParams = *g_searchParams;
            searchParams.maxVisits = maxVisits;
            searchParams.maxPlayouts = maxVisits;

            LOG_INFO(@"[%.3fs] Creating Search object...", timeMark());
            Search* search = new Search(searchParams, g_nnEval, g_logger, "iOSSearch");

            // 4. Set position
            LOG_INFO(@"[%.3fs] Setting position...", timeMark());
            search->setPosition(nextPla, board, history);

            // 5. Run search with progress tracking callbacks
            g_currentMaxVisits = maxVisits;
            g_shouldStop.store(false, std::memory_order_release);
            g_activeSearch = nullptr;  // Will be set in searchBegun

            std::function<void()> searchBegun = [&search]() {
                g_activeSearch = search;
                LOG_INFO(@"Search begun, progress tracking active");
            };
            std::function<bool()> shouldStopEarly = []() -> bool {
                return g_shouldStop.load(std::memory_order_acquire);
            };

            LOG_INFO(@"[%.3fs] Starting search (%d visits)...", timeMark(), maxVisits);
            TimeControls tc;
            search->runWholeSearch(&searchBegun, &shouldStopEarly, false, tc, 1.0);
            g_activeSearch = nullptr;  // Search done, clear pointer
            LOG_INFO(@"[%.3fs] Search completed", timeMark());

            // 6. Extract JSON results
            LOG_INFO(@"[%.3fs] Extracting JSON...", timeMark());
            json result;
            bool suc = search->getAnalysisJson(
                P_BLACK,   // perspective: always report from Black's perspective
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
                LOG_ERROR(@"getAnalysisJson failed, falling back to empty result");
                result["moveInfos"] = json::array();
                result["rootInfo"] = json::object();
            }

            result["id"] = "ios_analysis";
            result["turnNumber"] = (int)history.moveHistory.size();

            // 7. Cleanup
            LOG_INFO(@"[%.3fs] Deleting search...", timeMark());
            delete search;
            LOG_INFO(@"[%.3fs] Analysis complete", timeMark());

            // 8. Return JSON string
            std::string jsonStr = result.dump();
            return [NSString stringWithUTF8String:jsonStr.c_str()];

        } @catch (NSException *exception) {
            LOG_ERROR(@"Analysis failed: %@", exception.reason);
            return nil;
        }
    }
}

+ (void)destroy {
    @autoreleasepool {
        LOG_INFO(@"=== Destroying KataGo engine ===");

        if (g_nnEval != nullptr) {
            delete g_nnEval;
            g_nnEval = nullptr;
        }

        if (g_searchParams != nullptr) {
            delete g_searchParams;
            g_searchParams = nullptr;
        }

        if (g_logger != nullptr) {
            delete g_logger;
            g_logger = nullptr;
        }

        g_engineInitialized = false;
        LOG_INFO(@"✓ Engine destroyed");
    }
}

+ (BOOL)isInitialized {
    return g_engineInitialized;
}

+ (int64_t)getCurrentVisits {
    Search* search = g_activeSearch;
    if (search != nullptr) {
        return search->getRootVisits();
    }
    return 0;
}

+ (int64_t)getMaxVisits {
    return g_currentMaxVisits;
}

+ (void)requestStop {
    g_shouldStop.store(true, std::memory_order_release);
    LOG_INFO(@"Stop requested");
}

@end
