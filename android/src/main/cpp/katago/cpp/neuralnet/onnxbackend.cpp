#ifdef USE_ONNX_BACKEND

/** ONNX Runtime C++ Backend for Android
 *
 * Single-threaded ONNX Runtime backend to replace Eigen backend on Android.
 * Avoids pthread creation issues on certain devices (Snapdragon 8 Gen 3 + Android 16).
 *
 * Key features:
 * - Synchronous inference (no worker threads)
 * - NHWC -> NCHW format conversion for ONNX model
 * - Single-threaded mode (SetIntraOpNumThreads=1)
 */

#include "../neuralnet/nninterface.h"

#include <onnxruntime_cxx_api.h>

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"

#include "../core/global.h"
#include "../core/test.h"

#include <android/log.h>
#define TAG "KataGo-ONNX"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

using namespace std;

// ============================================================================
// LoadedModel / ModelDesc
// ============================================================================

struct LoadedModel {
  ModelDesc modelDesc;       // Metadata from .bin.gz
  string onnxModelPath;      // Path to .onnx file for inference

  LoadedModel(const string& binGzFile, const string& onnxFile, const string& expectedSha256)
    : onnxModelPath(onnxFile) {
    ModelDesc::loadFromFileMaybeGZipped(binGzFile, modelDesc, expectedSha256);
    LOGI("Loaded model: %s", modelDesc.name.c_str());
    LOGI("Model version: %d", modelDesc.modelVersion);
    LOGI("Input channels: %d spatial, %d global",
         modelDesc.numInputChannels, modelDesc.numInputGlobalChannels);
    LOGI("ONNX model path: %s", onnxFile.c_str());
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

// Global variable set by native-lib.cpp
extern std::string g_onnxModelPath;

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  // For ONNX backend, we expect TWO files:
  // - file: the .bin.gz for metadata
  // - g_onnxModelPath: the .onnx model (set by initializeNative)

  string onnxPath;
  if (!g_onnxModelPath.empty()) {
    onnxPath = g_onnxModelPath;
    LOGI("Using ONNX model path from global: %s", onnxPath.c_str());
  } else {
    // Fallback: try to guess from .bin.gz path
    onnxPath = file;
    size_t pos = onnxPath.rfind(".bin.gz");
    if (pos != string::npos) {
      onnxPath.replace(pos, 7, ".onnx");
    }
    LOGI("Guessed ONNX model path: %s", onnxPath.c_str());
  }

  LoadedModel* loadedModel = new LoadedModel(file, onnxPath, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

// ============================================================================
// ComputeContext - holds Ort::Env (global singleton)
// ============================================================================

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  Ort::Env ortEnv;

  ComputeContext(int x, int y)
    : nnXLen(x),
      nnYLen(y),
      ortEnv(ORT_LOGGING_LEVEL_WARNING, "KataGo-ONNX") {
    LOGI("Created ONNX ComputeContext for %dx%d board", x, y);
  }

  ~ComputeContext() {
    LOGI("Destroyed ONNX ComputeContext");
  }

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;
};

// ============================================================================
// ComputeHandle - holds Ort::Session (per-thread, single-threaded mode)
// ============================================================================

struct ComputeHandle {
  ComputeContext* context;
  bool inputsUseNHWC;
  const ModelDesc& modelDesc;

  Ort::SessionOptions sessionOptions;
  unique_ptr<Ort::Session> session;

  // Pre-allocated buffers for NCHW conversion
  vector<float> spatialNCHW;  // NCHW format for ONNX
  vector<float> globalBuffer;

  // Input/output tensor names
  vector<const char*> inputNames;
  vector<string> outputNameStrings;  // Must persist for C strings
  vector<const char*> outputNames;

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(
    ComputeContext* ctx,
    const LoadedModel& loadedModel,
    int maxBatchSize,
    bool iNHWC
  ) : context(ctx),
      inputsUseNHWC(iNHWC),
      modelDesc(loadedModel.modelDesc)
  {
    LOGI("Creating ONNX ComputeHandle...");

    // Configure session options for single-threaded mode
    sessionOptions.SetIntraOpNumThreads(1);  // Critical: no thread pool
    sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    // Note: NNAPI provider requires additional setup and may not be available in all ONNX Runtime builds
    // For now, we use CPU execution provider which is always available
    LOGI("Using CPU execution provider (single-threaded)");

    // Load ONNX model
    try {
      session = make_unique<Ort::Session>(
        ctx->ortEnv,
        loadedModel.onnxModelPath.c_str(),
        sessionOptions
      );
      LOGI("ONNX session created successfully");
    } catch (const Ort::Exception& e) {
      LOGE("Failed to create ONNX session: %s", e.what());
      throw StringError(string("ONNX session creation failed: ") + e.what());
    }

    // Pre-allocate conversion buffers
    const int C = modelDesc.numInputChannels;  // 22
    const int H = ctx->nnYLen;
    const int W = ctx->nnXLen;
    spatialNCHW.resize(maxBatchSize * C * H * W);
    globalBuffer.resize(maxBatchSize * modelDesc.numInputGlobalChannels);

    // Setup input/output names (must match ONNX model exactly)
    // Based on Dart onnx_engine.dart line 145
    inputNames = {"input_binary", "input_global"};

    // Query model for output names
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numOutputs = session->GetOutputCount();
    LOGI("ONNX model has %zu outputs", numOutputs);
    for (size_t i = 0; i < numOutputs; i++) {
      auto nameAlloc = session->GetOutputNameAllocated(i, allocator);
      outputNameStrings.push_back(string(nameAlloc.get()));
      LOGI("  Output %zu: %s", i, nameAlloc.get());
    }
    // Convert to const char* array
    for (const auto& s : outputNameStrings) {
      outputNames.push_back(s.c_str());
    }

    LOGI("ComputeHandle ready: maxBatch=%d, spatial=%dx%dx%d, global=%d",
         maxBatchSize, C, H, W, modelDesc.numInputGlobalChannels);
  }

  ~ComputeHandle() {
    LOGI("Destroyed ONNX ComputeHandle");
  }
};

// ============================================================================
// InputBuffers - same as Eigen backend
// ============================================================================

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputGlobalElts;
  size_t singleInputMetaElts;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyResultElts;
  size_t singleValueResultElts;
  size_t singleScoreValueResultElts;
  size_t singleOwnershipResultElts;

  vector<float> spatialInput;  // NHWC format (from KataGo feature encoding)
  vector<float> globalInput;
  vector<float> metaInput;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnXLen, int nnYLen) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = m.numInputChannels * nnXLen * nnYLen;
    singleInputGlobalElts = m.numInputGlobalChannels;
    singleInputMetaElts = m.numInputMetaChannels;

    singlePolicyPassResultElts = (size_t)(m.numPolicyChannels);
    singlePolicyResultElts = (size_t)(m.numPolicyChannels * nnXLen * nnYLen);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnXLen * nnYLen;

    assert(NNModelVersion::getNumSpatialFeatures(m.modelVersion) == m.numInputChannels);
    assert(NNModelVersion::getNumGlobalFeatures(m.modelVersion) == m.numInputGlobalChannels);
    if(m.numInputMetaChannels > 0) {
      assert(SGFMetadata::METADATA_INPUT_NUM_CHANNELS == m.numInputMetaChannels);
    }

    spatialInput = vector<float>(m.numInputChannels * nnXLen * nnYLen * maxBatchSize);
    globalInput = vector<float>(m.numInputGlobalChannels * maxBatchSize);
    if(m.numInputMetaChannels > 0)
      metaInput = vector<float>(m.numInputMetaChannels * maxBatchSize);
    else
      metaInput = vector<float>(1);
  }

  ~InputBuffers() { }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

// ============================================================================
// NHWC -> NCHW Conversion Helper
// ============================================================================

// Convert from KataGo's NHWC format (C varies fastest in memory) to
// ONNX's NCHW format (W varies fastest).
//
// NHWC memory layout (Eigen column-major [C,W,H,N]):
//   index = c + w*C + h*C*W + n*C*W*H
//
// NCHW memory layout:
//   index = n*C*H*W + c*H*W + h*W + w
static void convertNHWCtoNCHW(
  const float* nhwc,  // Input in NHWC
  float* nchw,        // Output in NCHW
  int N, int C, int H, int W
) {
  for (int n = 0; n < N; n++) {
    for (int c = 0; c < C; c++) {
      for (int h = 0; h < H; h++) {
        for (int w = 0; w < W; w++) {
          int nhwc_idx = n * C * H * W + c + w * C + h * C * W;
          int nchw_idx = n * C * H * W + c * H * W + h * W + w;
          nchw[nchw_idx] = nhwc[nhwc_idx];
        }
      }
    }
  }
}

// ============================================================================
// NeuralNet Interface Implementation
// ============================================================================

void NeuralNet::globalInitialize() {
  LOGI("ONNX Runtime backend: globalInitialize()");
  // ONNX Runtime handles initialization internally
}

void NeuralNet::globalCleanup() {
  LOGI("ONNX Runtime backend: globalCleanup()");
  // ONNX Runtime handles cleanup internally
}

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)gpuIdxs;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  if (useFP16Mode == enabled_t::True) {
    if (logger) logger->write("ONNX backend: FP16 not supported, using FP32");
  }

  bool useNHWC = useNHWCMode == enabled_t::False ? false : true;
  if (!useNHWC) {
    throw StringError("ONNX backend: useNHWC = false not supported");
  }

  ComputeContext* context = new ComputeContext(nnXLen, nnYLen);

  if (logger) {
    logger->write("ONNX Runtime backend initialized");
    logger->write("Board size: " + Global::intToString(nnXLen) + "x" + Global::intToString(nnYLen));
  }

  return context;
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  if (logger != NULL) {
    logger->write("ONNX Runtime backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model version " + Global::intToString(loadedModel->modelDesc.modelVersion));
    logger->write("ONNX Runtime backend thread " + Global::intToString(serverThreadIdx) +
                  ": Model name: " + loadedModel->modelDesc.name);
    logger->write("ONNX Runtime backend: Single-threaded mode (no pthread)");
  }

  (void)requireExactNNLen;
  (void)gpuIdxForThisThread;

  if (!inputsUseNHWC) {
    throw StringError("ONNX backend: inputsUseNHWC = false unsupported");
  }

  return new ComputeHandle(context, *loadedModel, maxBatchSize, inputsUseNHWC);
}

void NeuralNet::freeComputeHandle(ComputeHandle* handle) {
  delete handle;
}

bool NeuralNet::isUsingFP16(const ComputeHandle* handle) {
  (void)handle;
  return false;  // ONNX backend uses FP32
}

// ============================================================================
// Main Inference Function: getOutput()
// ============================================================================

void NeuralNet::getOutput(
  ComputeHandle* computeHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  assert(numBatchEltsFilled > 0);

  const int batchSize = numBatchEltsFilled;
  const int nnXLen = computeHandle->context->nnXLen;
  const int nnYLen = computeHandle->context->nnYLen;
  const int modelVersion = computeHandle->modelDesc.modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = inputBuffers->singleInputMetaElts;

  assert(numSpatialFeatures == computeHandle->modelDesc.numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);

  const int numPolicyChannels = computeHandle->modelDesc.numPolicyChannels;

  // Step 1: Copy inputs with symmetry transform (same as Eigen backend)
  for (int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->spatialInput.data() + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->globalInput.data() + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->metaInput.data() + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    const bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);

    if (numMetaFeatures > 0) {
      testAssert(rowMeta != NULL);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    } else {
      testAssert(!hasRowMeta);
    }

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures,
      computeHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry
    );
  }

  // Step 2: Convert NHWC -> NCHW for ONNX
  convertNHWCtoNCHW(
    inputBuffers->spatialInput.data(),
    computeHandle->spatialNCHW.data(),
    batchSize, numSpatialFeatures, nnYLen, nnXLen
  );

  std::copy(
    inputBuffers->globalInput.data(),
    inputBuffers->globalInput.data() + batchSize * numGlobalFeatures,
    computeHandle->globalBuffer.data()
  );

  // Step 3: Create ONNX input tensors
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  // Spatial input: [N, C, H, W]
  vector<int64_t> spatialShape = {batchSize, numSpatialFeatures, nnYLen, nnXLen};
  Ort::Value spatialTensor = Ort::Value::CreateTensor<float>(
    memoryInfo,
    computeHandle->spatialNCHW.data(),
    batchSize * numSpatialFeatures * nnYLen * nnXLen,
    spatialShape.data(),
    spatialShape.size()
  );

  // Global input: [N, G]
  vector<int64_t> globalShape = {batchSize, numGlobalFeatures};
  Ort::Value globalTensor = Ort::Value::CreateTensor<float>(
    memoryInfo,
    computeHandle->globalBuffer.data(),
    batchSize * numGlobalFeatures,
    globalShape.data(),
    globalShape.size()
  );

  vector<Ort::Value> inputTensors;
  inputTensors.push_back(std::move(spatialTensor));
  inputTensors.push_back(std::move(globalTensor));

  // Step 4: Run ONNX inference (synchronous, no threads)
  try {
    // Run with explicit output names
    vector<Ort::Value> outputTensors = computeHandle->session->Run(
      Ort::RunOptions{nullptr},
      computeHandle->inputNames.data(),
      inputTensors.data(),
      inputTensors.size(),
      computeHandle->outputNames.data(),
      computeHandle->outputNames.size()
    );

    // Step 5: Extract output tensor data
    // outputs[0] = output_policy: [1, boardSize*boardSize+1]
    // outputs[1] = output_value: [1, 3]
    // outputs[2] = output_miscvalue: [1, 4 or 6]
    // outputs[3] = output_ownership: [1, boardSize*boardSize]

    if (outputTensors.size() < 2) {
      throw StringError("ONNX model returned < 2 outputs");
    }

    // Get raw pointers to output data
    float* policyData = outputTensors[0].GetTensorMutableData<float>();
    float* valueData = outputTensors[1].GetTensorMutableData<float>();

    float* scoreValueData = outputTensors.size() > 2
        ? outputTensors[2].GetTensorMutableData<float>()
        : valueData;  // Fallback
    float* ownershipData = outputTensors.size() > 3
        ? outputTensors[3].GetTensorMutableData<float>()
        : nullptr;

    // Determine actual number of policy channels from ONNX output tensor size.
    // The .bin.gz model descriptor may say numPolicyChannels=2, but the ONNX
    // export may flatten the policy to a single vector [1, H*W+1].
    auto policyTensorInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    size_t policyTotalElts = policyTensorInfo.GetElementCount() / std::max(batchSize, 1);
    int onnxPolicyChannels;
    if ((int)policyTotalElts == nnXLen * nnYLen + 1) {
      onnxPolicyChannels = 1;  // Flat output: [N, H*W+1]
    } else {
      onnxPolicyChannels = numPolicyChannels;  // Multi-channel: [N, C, H, W] + pass
    }

    // Policy includes pass as last element (per channel)
    float* policyPassData = policyData + (nnXLen * nnYLen) * onnxPolicyChannels;

    // Step 6: Fill NNOutput structs (same logic as Eigen backend)
    float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

    assert(outputs.size() == batchSize);

    for (int row = 0; row < batchSize; row++) {
      NNOutput* output = outputs[row];
      assert(output->nnXLen == nnXLen);
      assert(output->nnYLen == nnYLen);
      float policyOptimism = (float)inputBufs[row]->policyOptimism;

      const float* policyPassSrcBuf = policyPassData + row * onnxPolicyChannels;
      const float* policySrcBuf = policyData + row * onnxPolicyChannels * nnXLen * nnYLen;
      float* policyProbs = output->policyProbs;

      // Set policyOptimismUsed
      output->policyOptimismUsed = policyOptimism;

      // Policy logits (not probabilities yet - client does softmax)
      // Use onnxPolicyChannels (derived from actual ONNX output shape) instead
      // of numPolicyChannels (from .bin.gz model descriptor) to avoid reading
      // out-of-bounds when the ONNX export flattened the policy channels.
      if (onnxPolicyChannels >= 2) {
        // Multi-channel ONNX output: [N, C=2, H, W] + pass
        for (int h = 0; h < nnYLen; h++) {
          for (int w = 0; w < nnXLen; w++) {
            int i = h * nnXLen + w;
            float p = policySrcBuf[0 * nnYLen * nnXLen + i];
            float pOpt = policySrcBuf[1 * nnYLen * nnXLen + i];
            policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
          }
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry
        );
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] +
                                       (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      } else {
        // Flat ONNX output: [N, H*W+1] — single policy channel
        for (int i = 0; i < nnXLen * nnYLen; i++) {
          policyProbsTmp[i] = policySrcBuf[i];
        }
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry
        );
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
      }

      // Value logits (win/loss/noResult)
      int numValueChannels = computeHandle->modelDesc.numValueChannels;
      assert(numValueChannels == 3);
      output->whiteWinProb = valueData[row * numValueChannels];
      output->whiteLossProb = valueData[row * numValueChannels + 1];
      output->whiteNoResultProb = valueData[row * numValueChannels + 2];

      // Ownership (if requested)
      if (output->whiteOwnerMap != NULL && ownershipData != NULL) {
        const float* ownershipSrcBuf = ownershipData + row * nnXLen * nnYLen;
        // ONNX ownership output: [N, 1, H, W] in NCHW
        // Convert to flat for SymmetryHelpers
        for (int i = 0; i < nnXLen * nnYLen; i++) {
          policyProbsTmp[i] = ownershipSrcBuf[i];
        }
        assert(computeHandle->modelDesc.numOwnershipChannels == 1);
        SymmetryHelpers::copyOutputsWithSymmetry(
          policyProbsTmp, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry
        );
      }

      // Score value from output_miscvalue tensor
      {
        int numScoreValueChannels = computeHandle->modelDesc.numScoreValueChannels;
        int modelVersion = computeHandle->modelDesc.modelVersion;
        if (modelVersion >= 9 && numScoreValueChannels >= 6) {
          output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
          output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
          output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
          output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
          output->shorttermWinlossError = scoreValueData[row * numScoreValueChannels + 4];
          output->shorttermScoreError = scoreValueData[row * numScoreValueChannels + 5];
        } else if (modelVersion >= 8 && numScoreValueChannels >= 4) {
          output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
          output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
          output->whiteLead = scoreValueData[row * numScoreValueChannels + 2];
          output->varTimeLeft = scoreValueData[row * numScoreValueChannels + 3];
          output->shorttermWinlossError = 0;
          output->shorttermScoreError = 0;
        } else if (modelVersion >= 4 && numScoreValueChannels >= 2) {
          output->whiteScoreMean = scoreValueData[row * numScoreValueChannels];
          output->whiteScoreMeanSq = scoreValueData[row * numScoreValueChannels + 1];
          output->whiteLead = output->whiteScoreMean;
          output->varTimeLeft = 0;
          output->shorttermWinlossError = 0;
          output->shorttermScoreError = 0;
        } else {
          // Fallback for very old models
          output->whiteScoreMean = 0.0f;
          output->whiteScoreMeanSq = 1.0f;
          output->whiteLead = 0.0f;
          output->varTimeLeft = 1.0f;
          output->shorttermWinlossError = 0.1f;
          output->shorttermScoreError = 0.1f;
        }
      }
    }

    LOGI("ONNX inference completed for batch size %d", batchSize);

  } catch (const Ort::Exception& e) {
    LOGE("ONNX inference failed: %s", e.what());
    throw StringError(string("ONNX inference error: ") + e.what());
  }
}

// ============================================================================
// Test Functions (Not Implemented)
// ============================================================================

void NeuralNet::printDevices() {
  LOGI("ONNX Runtime backend: CPU only (single-threaded)");
}

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer; (void)outputBuffer;
  return false;  // Not implemented for ONNX backend
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer;
  (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer;
  (void)maskBuffer; (void)outputBuffer;
  return false;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const std::vector<float>& inputBuffer,
  const std::vector<float>& maskBuffer,
  std::vector<float>& outputBuffer
) {
  (void)desc; (void)batchSize; (void)nnXLen; (void)nnYLen;
  (void)useFP16; (void)useNHWC; (void)inputBuffer;
  (void)maskBuffer; (void)outputBuffer;
  return false;
}

#endif // USE_ONNX_BACKEND
