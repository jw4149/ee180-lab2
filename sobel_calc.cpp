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
  Mat img_outx = img_gray.clone();
  Mat img_outy = img_gray.clone();
  img_sobel_out = img_gray.clone();

  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j += 8) {
      uint8x8_t one = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j-1));
      int16x8_t one16 = vreinterpretq_s16_u16(vmovl_u8(one));

      uint8x8_t two = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j-1));
      int16x8_t two16 = vreinterpretq_s16_u16(vmovl_u8(two));

      uint8x8_t three = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j));
      int16x8_t three16 = vreinterpretq_s16_u16(vmovl_u8(three));
      three16 = vaddq_s16(three16, three16);

      uint8x8_t four = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j));
      int16x8_t four16 = vreinterpretq_s16_u16(vmovl_u8(four));
      four16 = vaddq_s16(four16, four16);

      uint8x8_t five = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j+1));
      int16x8_t five16 = vreinterpretq_s16_u16(vmovl_u8(five));

      uint8x8_t six = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j+1));
      int16x8_t six16 = vreinterpretq_s16_u16(vmovl_u8(six));

      int16x8_t sobel16 = vsubq_s16(one16, two16);
      sobel16 = vaddq_s16(sobel16, three16);
      sobel16 = vsubq_s16(sobel16, four16);
      sobel16 = vaddq_s16(sobel16, five16);
      sobel16 = vsubq_s16(sobel16, six16);

      sobel16 = vabsq_s16(sobel16);
      sobel16 = vminq_s16(sobel16, vdupq_n_s16(255));

      vst1_u8(img_outx.data+IMG_WIDTH*(i)+(j), vreinterpret_u8_s8(vmovn_s16(sobel16)));
    }
  }

  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j += 8) {
      uint8x8_t one = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j-1));
      int16x8_t one16 = vreinterpretq_s16_u16(vmovl_u8(one));

      uint8x8_t two = vld1_u8(img_gray.data+IMG_WIDTH*(i-1)+(j+1));
      int16x8_t two16 = vreinterpretq_s16_u16(vmovl_u8(two));

      uint8x8_t three = vld1_u8(img_gray.data+IMG_WIDTH*(i)+(j-1));
      int16x8_t three16 = vreinterpretq_s16_u16(vmovl_u8(three));
      three16 = vaddq_s16(three16, three16);

      uint8x8_t four = vld1_u8(img_gray.data+IMG_WIDTH*(i)+(j+1));
      int16x8_t four16 = vreinterpretq_s16_u16(vmovl_u8(four));
      four16 = vaddq_s16(four16, four16);

      uint8x8_t five = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j-1));
      int16x8_t five16 = vreinterpretq_s16_u16(vmovl_u8(five));

      uint8x8_t six = vld1_u8(img_gray.data+IMG_WIDTH*(i+1)+(j+1));
      int16x8_t six16 = vreinterpretq_s16_u16(vmovl_u8(six));

      int16x8_t sobel16 = vsubq_s16(one16, two16);
      sobel16 = vaddq_s16(sobel16, three16);
      sobel16 = vsubq_s16(sobel16, four16);
      sobel16 = vaddq_s16(sobel16, five16);
      sobel16 = vsubq_s16(sobel16, six16);

      sobel16 = vabsq_s16(sobel16);
      sobel16 = vminq_s16(sobel16, vdupq_n_s16(255));

      vst1_u8(img_outy.data+IMG_WIDTH*(i)+(j), vreinterpret_u8_s8(vmovn_s16(sobel16)));
    }
  }

  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j += 8) {

      uint8x8_t xdata = vld1_u8(img_outx.data+IMG_WIDTH*(i)+(j));
      uint8x8_t ydata = vld1_u8(img_outy.data+IMG_WIDTH*(i)+(j));

      uint16x8_t xdata16 = vmovl_u8(xdata);
      uint16x8_t ydata16 = vmovl_u8(ydata);

      uint16x8_t sobel = vaddq_u16(xdata16, ydata16);
      sobel = vminq_u16(sobel, vdupq_n_u16(255));

      vst1_u8(img_sobel_out.data+IMG_WIDTH*(i)+(j), vmovn_u16(sobel));

    }
  }
}
