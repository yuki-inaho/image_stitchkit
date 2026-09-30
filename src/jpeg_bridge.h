#ifndef STITCHKIT_JPEG_BRIDGE_H
#define STITCHKIT_JPEG_BRIDGE_H
#include <stdio.h>
#ifdef __cplusplus
extern "C" {
#endif
unsigned char* stitchkit_decode_jpeg(FILE* file, int* width, int* height, char error[256]);
#ifdef __cplusplus
}
#endif
#endif
