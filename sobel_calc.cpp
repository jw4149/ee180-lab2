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
  // double color;

  unsigned img_len = img.rows * img.cols;
  
  float32x4x3_t intlv_rgb;
  for (int i = 0; i < img_len/4; i++) {
    intlv_rgb = vld3q_f32(img.data + 3*4*i);
    float32x4_t r_mult_coeff = vdupq_n_f32(.114f);
    float32x4_t g_mult_coeff = vdupq_n_f32(.587f);
    float32x4_t b_mult_coeff = vdupq_n_f32(.299f);
    intlv_rgb.val[0] = vmulq_f32(intlv_rgb.val[0], r_mult_coeff);
    intlv_rgb.val[1] = vmulq_f32(intlv_rgb.val[1], g_mult_coeff);
    intlv_rgb.val[2] = vmulq_f32(intlv_rgb.val[2], b_mult_coeff);
    float32x4_t color = vaddq_f32(intlv_rgb.val[0], intlv_rgb.val[1]);
    color = vaddq_f32(color, intlv_rgb.val[2]);
    vst3q_f32(img_gray_out.data + 3*4*i, color);
  }


  // Convert to grayscale
  // for (int i=0; i<img.rows; i++) {
  //   for (int j=0; j<img.cols; j++) {
  //     color = .114*img.data[STEP0*i + STEP1*j] +
  //             .587*img.data[STEP0*i + STEP1*j + 1] +
  //             .299*img.data[STEP0*i + STEP1*j + 2];
  //     img_gray_out.data[IMG_WIDTH*i + j] = color;
  //   }
  // }
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
