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

  // Apply Sobel filter to black & white image
  unsigned short sobel;

  // Calculate the x convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j-1)] +
		  2*img_gray.data[IMG_WIDTH*(i-1) + (j)] -
		  2*img_gray.data[IMG_WIDTH*(i+1) + (j)] +
		  img_gray.data[IMG_WIDTH*(i-1) + (j+1)] -
		  img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

      sobel = (sobel > 255) ? 255 : sobel;
      img_outx.data[IMG_WIDTH*(i) + (j)] = sobel;
    }
  }

  // Calc the y convolution
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
     sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

     sobel = (sobel > 255) ? 255 : sobel;

     img_outy.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }

  // Combine the two convolutions into the output image
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel = img_outx.data[IMG_WIDTH*(i) + j] +
	            img_outy.data[IMG_WIDTH*(i) + j];
      sobel = (sobel > 255) ? 255 : sobel;
      img_sobel_out.data[IMG_WIDTH*(i) + j] = sobel;
    }
  }
}
