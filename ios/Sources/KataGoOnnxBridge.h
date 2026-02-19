//
//  KataGoOnnxBridge.h
//  KataGoMobile
//
//  Created for Go Strategy App
//  ONNX Runtime-based KataGo analysis bridge for iOS
//

#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface KataGoOnnxBridge : NSObject

/**
 * Initialize the ONNX-based KataGo engine
 *
 * @param configPath Path to analysis.cfg file
 * @param modelBinPath Path to model.bin.gz file
 * @param modelOnnxPath Path to board-size-specific .onnx model
 * @param boardSize Board size (9, 13, or 19)
 * @return YES if initialization succeeds, NO otherwise
 */
+ (BOOL)initializeWithConfig:(NSString *)configPath
                    modelBin:(NSString *)modelBinPath
                   modelOnnx:(NSString *)modelOnnxPath
                   boardSize:(int)boardSize;

/**
 * Analyze a Go position using KataGo's MCTS search
 *
 * @param params Dictionary containing:
 *   - boardXSize: int (board width)
 *   - boardYSize: int (board height)
 *   - komi: double (komi value)
 *   - maxVisits: int (MCTS visits)
 *   - moves: NSArray of NSArray (each move is @[@"B", @"Q16"] format)
 * @return JSON string with analysis results, or nil on error
 */
+ (nullable NSString *)analyzePosition:(NSDictionary *)params;

/**
 * Clean up and destroy the engine
 */
+ (void)destroy;

/**
 * Check if engine is initialized
 */
+ (BOOL)isInitialized;

/**
 * Get current search visits (thread-safe, atomic read)
 * Returns 0 if no search is in progress
 */
+ (int64_t)getCurrentVisits;

/**
 * Get max visits for the current analysis
 */
+ (int64_t)getMaxVisits;

/**
 * Request the current search to stop early
 * Search will stop after the next playout iteration
 */
+ (void)requestStop;

@end

NS_ASSUME_NONNULL_END
