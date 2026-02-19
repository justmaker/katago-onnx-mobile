#include "katago/cpp/main.h"
#include <string>
#include <vector>

namespace Version {
std::string getKataGoVersion() { return "1.15.0"; }
std::string getKataGoVersionForHelp() { return "KataGo v1.15.0"; }
std::string getKataGoVersionFullInfo() { return "KataGo v1.15.0 (Mobile)"; }
std::string getGitRevision() { return "unknown"; }
std::string getGitRevisionWithBackend() { return "unknown-eigen"; }
} // namespace Version

// Stubs for Tests
namespace TestCommon {
std::string getBenchmarkSGFData(int) { return ""; }
} // namespace TestCommon

// Stubs for training data writing (not needed in mobile)
// We provide these because some core files might link to them
void recordTreePositionsRec(...) {}

// If we exclude trainingwrite.cpp, we might need these:
/*
namespace Play {
    // ...
}
*/

// For now, let's try to include trainingwrite.cpp but mock zip.h
