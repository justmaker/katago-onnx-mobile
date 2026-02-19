#ifndef FAKE_ZIP_H
#define FAKE_ZIP_H

#include <stdint.h>

typedef void zip_t;
typedef void zip_source_t;
typedef struct {
  int dummy;
} zip_error_t;
typedef int64_t zip_int64_t;

#define ZIP_CREATE 1
#define ZIP_EXCL 2
#define ZIP_CHECKCONS 4
#define ZIP_TRUNCATE 8

#define ZIP_FL_OVERWRITE 8192

inline zip_t *zip_open(const char *, int, int *) { return nullptr; }
inline int zip_close(zip_t *) { return 0; }
inline void zip_discard(zip_t *) {}
inline zip_source_t *zip_source_buffer(zip_t *, const void *, uint64_t, int) {
  return nullptr;
}
inline int64_t zip_file_add(zip_t *, const char *, zip_source_t *, uint32_t) {
  return 0;
}
inline void zip_source_free(zip_source_t *) {}
inline void zip_error_init(zip_error_t *) {}
inline void zip_error_fini(zip_error_t *) {}
inline zip_source_t *zip_source_file_create(const char *, uint64_t, int64_t,
                                            zip_error_t *) {
  return nullptr;
}
inline const char *zip_error_strerror(zip_error_t *) { return "stub"; }
inline zip_t *zip_open_from_source(zip_source_t *, int, zip_error_t *) {
  return nullptr;
}
inline int zip_error_code_zip(const zip_error_t *) { return 0; }
inline const char *zip_strerror(zip_t *) { return "stub"; }

#endif
