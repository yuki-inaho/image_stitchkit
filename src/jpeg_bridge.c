#include "jpeg_bridge.h"
#include <jpeglib.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
struct DecodeState {
    struct jpeg_decompress_struct decoder;
    struct jpeg_error_mgr errors;
    jmp_buf jump;
    unsigned char* pixels;
    char message[256];
};
static void fail(j_common_ptr common) {
    struct DecodeState* state = (struct DecodeState*)common->client_data;
    (*common->err->format_message)(common, state->message);
    longjmp(state->jump, 1);
}
unsigned char* stitchkit_decode_jpeg(FILE* file, int* width, int* height, char error[256]) {
    struct DecodeState* state = (struct DecodeState*)calloc(1, sizeof(struct DecodeState));
    unsigned char* result;
    if (!state) {
        strcpy(error, "JPEG state allocation failed");
        return NULL;
    }
    state->decoder.err = jpeg_std_error(&state->errors);
    state->errors.error_exit = fail;
    state->decoder.client_data = state;
    if (setjmp(state->jump)) {
        strncpy(error, state->message, 255);
        error[255] = 0;
        jpeg_destroy_decompress(&state->decoder);
        free(state->pixels);
        free(state);
        return NULL;
    }
    jpeg_create_decompress(&state->decoder);
    state->decoder.client_data = state;
    jpeg_stdio_src(&state->decoder, file);
    jpeg_read_header(&state->decoder, TRUE);
    if (state->decoder.image_width < 2 || state->decoder.image_height < 2 ||
        state->decoder.image_width > 32768 || state->decoder.image_height > 32768 ||
        (uint64_t)state->decoder.image_width * state->decoder.image_height > 64000000) {
        strcpy(error, "Invalid or oversized JPEG dimensions");
        jpeg_destroy_decompress(&state->decoder);
        free(state);
        return NULL;
    }
    state->decoder.out_color_space = JCS_RGB;
    jpeg_start_decompress(&state->decoder);
    *width = (int)state->decoder.output_width;
    *height = (int)state->decoder.output_height;
    state->pixels = (unsigned char*)malloc((size_t)*width * *height * 3);
    if (!state->pixels) {
        strcpy(error, "JPEG pixel allocation failed");
        jpeg_destroy_decompress(&state->decoder);
        free(state);
        return NULL;
    }
    while (state->decoder.output_scanline < state->decoder.output_height) {
        JSAMPROW row = state->pixels + (size_t)state->decoder.output_scanline * *width * 3;
        jpeg_read_scanlines(&state->decoder, &row, 1);
    }
    jpeg_finish_decompress(&state->decoder);
    jpeg_destroy_decompress(&state->decoder);
    result = state->pixels;
    free(state);
    return result;
}
