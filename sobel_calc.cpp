#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include "arm_neon.h"
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  unsigned img_len = img.rows * img.cols;
  
  uint8x8x3_t rgbs;
  for (unsigned i = 0; i < img_len/8; i++) {

    rgbs = vld3_u8(img.data+i*3*8);

    uint16x8_t rs = vmovl_u8(rgbs.val[0]);
    uint16x8_t gs = vmovl_u8(rgbs.val[1]);
    uint16x8_t bs = vmovl_u8(rgbs.val[2]);

    rs = vmulq_n_u16(rs, 29);
    gs = vmulq_n_u16(gs, 150);
    bs = vmulq_n_u16(bs, 76);

    rs = vshrq_n_u16(rs, 8);
    gs = vshrq_n_u16(gs, 8);
    bs = vshrq_n_u16(bs, 8);

    uint16x8_t color = vaddq_u16(rs, gs);
    color = vaddq_u16(color, bs);

    vst1_u8(img_gray_out.data+i*8, vmovn_u16(color));
  }
}

/*******************************************
 * Model: sobelCalc
 * Input: Mat img_in
 * Output: None directly. Modifies a ref parameter img_sobel_out
 * Desc: This module performs a sobel calculation on an image. It first
 *  converts the image to grayscale, calculates the gradient in the x
 *  direction, calculates the gradient in the y direction and sum it with Gx
 *  to finish the Sobel calculation
 ********************************************/
void sobelCalc(Mat& img_gray, Mat& img_sobel_out)
{
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=0; j<img_gray.cols; j += 8) {
      uint8x8_t upper_left = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j));
      int16x8_t upper_left16 = vreinterpretq_s16_u16(vmovl_u8(upper_left));

      uint8x8_t lower_left = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j));
      int16x8_t lower_left16 = vreinterpretq_s16_u16(vmovl_u8(lower_left));

      uint8x8_t upper = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j+1));
      int16x8_t upper16 = vreinterpretq_s16_u16(vmovl_u8(upper));
      upper16 = vaddq_s16(upper16, upper16);

      uint8x8_t lower = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j+1));
      int16x8_t lower16 = vreinterpretq_s16_u16(vmovl_u8(lower));
      lower16 = vaddq_s16(lower16, lower16);

      uint8x8_t left = vld1_u8(img_gray.data+IMG_WIDTH*(i)+(j));
      int16x8_t left16 = vreinterpretq_s16_u16(vmovl_u8(left));
      left16 = vaddq_s16(left16, left16);

      uint8x8_t right = vld1_u8(img_gray.data+IMG_WIDTH*(i)+(j+2));
      int16x8_t right16 = vreinterpretq_s16_u16(vmovl_u8(right));
      right16 = vaddq_s16(right16, right16);

      uint8x8_t upper_right = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j+2));
      int16x8_t upper_right16 = vreinterpretq_s16_u16(vmovl_u8(upper_right));

      uint8x8_t lower_right = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j+2));
      int16x8_t lower_right16 = vreinterpretq_s16_u16(vmovl_u8(lower_right));

      int16x8_t sobel16x = vsubq_s16(upper_left16, lower_left16);
      sobel16x = vaddq_s16(sobel16x, upper16);
      sobel16x = vsubq_s16(sobel16x, lower16);
      sobel16x = vaddq_s16(sobel16x, upper_right16);
      sobel16x = vsubq_s16(sobel16x, lower_right16);

      sobel16x = vabsq_s16(sobel16x);
      sobel16x = vminq_s16(sobel16x, vdupq_n_s16(255));

      int16x8_t sobel16y = vsubq_s16(upper_left16, upper_right16);
      sobel16y = vaddq_s16(sobel16y, left16);
      sobel16y = vsubq_s16(sobel16y, right16);
      sobel16y = vaddq_s16(sobel16y, lower_left16);
      sobel16y = vsubq_s16(sobel16y, lower_right16);

      sobel16y = vabsq_s16(sobel16y);
      sobel16y = vminq_s16(sobel16y, vdupq_n_s16(255));

      uint16x8_t sobel16ux = vreinterpretq_u16_s16(sobel16x);
      uint16x8_t sobel16uy = vreinterpretq_u16_s16(sobel16y);

      uint16x8_t sobel = vaddq_u16(sobel16ux, sobel16uy);
      sobel = vminq_u16(sobel, vdupq_n_u16(255));

      vst1_u8(img_sobel_out.data+IMG_WIDTH*(i)+(j), vmovn_u16(sobel));
    }
  }
}
