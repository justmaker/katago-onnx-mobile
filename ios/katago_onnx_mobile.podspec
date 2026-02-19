Pod::Spec.new do |s|
  s.name             = 'katago_onnx_mobile'
  s.version          = '0.1.0'
  s.summary          = 'Flutter plugin for KataGo ONNX inference on iOS'
  s.description      = <<-DESC
KataGo Go AI engine with ONNX Runtime backend for on-device inference on iOS.
                       DESC
  s.homepage         = 'https://github.com/justmaker/katago-onnx-mobile'
  s.license          = { :file => '../LICENSE' }
  s.author           = { 'Rex Hsu' => 'justmaker@gmail.com' }
  s.source           = { :path => '.' }

  s.source_files = [
    'Classes/**/*.{h,m,mm,cpp,swift}',
    'Sources/*.{h,mm,cpp}',
    'Sources/katago/cpp/**/*.{h,cpp}',
    'Sources/katago/cpp/neuralnet/onnxbackend.cpp'
  ]
  s.public_header_files = 'Sources/KataGoOnnxBridge.h'
  s.libraries = 'z'

  s.dependency 'Flutter'
  s.dependency 'onnxruntime-objc', '~> 1.15.1'

  # Exclude non-iOS backends and tests
  s.exclude_files = [
    'Sources/katago/cpp/main.cpp',
    'Sources/katago/cpp/tests/**/*',
    'Sources/katago/cpp/neuralnet/opencl*',
    'Sources/katago/cpp/neuralnet/cuda*',
    'Sources/katago/cpp/neuralnet/tensorrt*',
    'Sources/katago/cpp/neuralnet/eigenbackend*',
    'Sources/katago/cpp/neuralnet/dummybackend*',
    'Sources/katago/cpp/distributed/**/*',
    'Sources/katago/cpp/command/benchmark.cpp',
    'Sources/katago/cpp/command/contribute.cpp',
    'Sources/katago/cpp/command/demoplay.cpp',
    'Sources/katago/cpp/command/evalsgf.cpp',
    'Sources/katago/cpp/command/gatekeeper.cpp',
    'Sources/katago/cpp/command/genbook.cpp',
    'Sources/katago/cpp/command/gputest.cpp',
    'Sources/katago/cpp/command/match.cpp',
    'Sources/katago/cpp/command/misc.cpp',
    'Sources/katago/cpp/command/runtests.cpp',
    'Sources/katago/cpp/command/sandbox.cpp',
    'Sources/katago/cpp/command/selfplay.cpp',
    'Sources/katago/cpp/command/startposes.cpp',
    'Sources/katago/cpp/command/tune.cpp',
    'Sources/katago/cpp/command/writetrainingdata.cpp'
  ]

  s.ios.deployment_target = '13.0'
  s.swift_version = '5.0'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386',
    'CLANG_CXX_LANGUAGE_STANDARD' => 'c++17',
    'CLANG_CXX_LIBRARY' => 'libc++',
    'GCC_PREPROCESSOR_DEFINITIONS' => 'USE_ONNX_BACKEND=1 NO_GIT_REVISION=1',
    'HEADER_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/Sources/eigen" "$(PODS_TARGET_SRCROOT)/Sources/katago/cpp" "$(PODS_TARGET_SRCROOT)/Sources/katago/cpp/external" "$(PODS_TARGET_SRCROOT)/Sources/katago/cpp/external/tclap-1.2.5/include" "$(PODS_TARGET_SRCROOT)/Sources/katago/cpp/external/nlohmann_json" "$(PODS_TARGET_SRCROOT)/Sources/katago/cpp/external/filesystem-1.5.8/include" "$(PODS_TARGET_SRCROOT)/Sources/fake_zip" "$(PODS_ROOT)/onnxruntime-c/onnxruntime.xcframework/ios-arm64/onnxruntime.framework/Headers" "$(PODS_ROOT)/onnxruntime-c/onnxruntime.xcframework/ios-arm64_x86_64-simulator/onnxruntime.framework/Headers"'
  }
end
