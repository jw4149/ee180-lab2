#include "opencv2/imgproc/imgproc.hpp"
#include "sobel_alg.h"
#include "arm_neon.h" /// added by me
using namespace cv;

/*******************************************
 * Model: grayScale
 * Input: Mat img
 * Output: None directly. Modifies a ref parameter img_gray_out
 * Desc: This module converts the image to grayscale
 ********************************************/
void grayScale(Mat& img, Mat& img_gray_out)
{
  double color;

  // Convert to grayscale
  for (int i=0; i<img.rows; i++) {
    for (int j=0; j<img.cols; j++) {
      /// Compute grayscale value using weighted average of RGB channels
      color = .114*img.data[STEP0*i + STEP1*j] +
              .587*img.data[STEP0*i + STEP1*j + 1] +
              .299*img.data[STEP0*i + STEP1*j + 2];
      img_gray_out.data[IMG_WIDTH*i + j] = color;
    }
  }
}

/********************** write by my own **********************/
void grayScale_opt(Mat& img, Mat& img_gray_out)
{
  float coefficients[] = {0.114, 0.587, 0.299};
  uint16_t coefficients_fixed[] = { (uint16_t)(256 * coefficients[0]), 
                                    (uint16_t)(256 * coefficients[1]), 
                                    (uint16_t)(256 * coefficients[2]) };
  uint16x4x3_t img_v;
  uint16x4_t img_gray;
  img_v = vld3_u16(img.data);
  for(int i = 0; i < img.rows; i++){
    for(int j = 0; j < img.cols; j++){
      img_gray = vmul_u16(coefficients_fixed[0]/256, img_v.val[0][STEP0*i + STEP1*j]) + 
                 vmul_u16(coefficients_fixed[1]/256, img_v.val[1][STEP0*i + STEP1*j]) +
                 vmul_u16(coefficients_fixed[2]/256, img_v.val[2][STEP0*i + STEP1*j]);
      vst1_u16(&img_gray_out.data[IMG_WIDTH*i + j], img_gray); // moving result back to normal C
    }
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

/********************** write by my own **********************/

void sobelCalc_opt(Mat& img_gray, Mat& img_sobel_out)
{
  Mat img_outx = img_gray.clone();
  Mat img_outy = img_gray.clone();

  // Apply Sobel filter to black & white image
  unsigned short sobel;

  // Calculate the x convolution
  uint16x4x3_t img_gray_vx;
  img_gray_vx = vld3_u16(img_gray.data);
  uint16x4_t sobel_add0; sobel_add1;
  for (int i=1; i<img_gray.rows; i++) {
      for (int j=1; j<img_gray.cols; j++) {

        sobel_add0 = vadd_u16(img_gray_vx.val[0][IMG_WIDTH*(i-1) + (j-1)], 2 * img_gray_vx.val[1][IMG_WIDTH*(i-1) + (j-1)]);
        sobel_add0 = vadd_u16(sobel_add0, img_gray_vx.val[2][IMG_WIDTH*(i-1) + (j-1)]);
        
        sobel_add1 = vadd_u16(img_gray_vx.val[0][IMG_WIDTH*(i+1) + (j-1)], 2 * img_gray_vx.val[1][IMG_WIDTH*(i+1) + (j-1)]);
        sobel_add1 = vadd_u16(sobel_add1, img_gray_vx.val[2][IMG_WIDTH*(i+1) + (j-1)]);

        sobel = vabd_u16(sobel_add0, sobel_add1); // subtract + abs

        sobel = (sobel > 255) ? 255 : sobel;
        vst1_u16(img_outx.data[IMG_WIDTH*i + j], sobel);
      }
    }
  

  // Calc the y convolution
  uint16x4x2_t img_gray_vy;
  img_gray_vx = vld2_u16(img_gray.data);
  uint16x4_t sobel_sub0; sobel_sub1, sobel_sub2;
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
     sobel = abs(img_gray.data[IMG_WIDTH*(i-1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i-1) + (j+1)] +
		   2*img_gray.data[IMG_WIDTH*(i) + (j-1)] -
		   2*img_gray.data[IMG_WIDTH*(i) + (j+1)] +
		   img_gray.data[IMG_WIDTH*(i+1) + (j-1)] -
		   img_gray.data[IMG_WIDTH*(i+1) + (j+1)]);

     sobel_sub0 = vsub_u16(img_gray_vx.val[0][IMG_WIDTH*(i-1) + (j-1)], img_gray_vx.val[1][IMG_WIDTH*(i-1) + (j-1)]);
     sobel_sub1 = vsub_u16(img_gray_vx.val[0][IMG_WIDTH*(i-1) + (j)], img_gray_vx.val[1][IMG_WIDTH*(i-1) + (j)]);
     sobel_sub2 = vsub_u16(img_gray_vx.val[0][IMG_WIDTH*(i-1) + (j+1)], img_gray_vx.val[1][IMG_WIDTH*(i-1) + (j+1)]);

     sobel = vadd_u16(sobel_sub0, sobel_sub1);
     sobel = vabs_u16(vadd_u16(sobel, sobel_sub2));

     sobel = (sobel > 255) ? 255 : sobel;

     vst1_u16(img_outy.data[IMG_WIDTH*i + j], sobel);
    }
  }

  // Combine the two convolutions into the output image
  for (int i=1; i<img_gray.rows; i++) {
    for (int j=1; j<img_gray.cols; j++) {
      sobel = img_outx.data[IMG_WIDTH*(i) + j] +
	img_outy.data[IMG_WIDTH*(i) + j];
      sobel = (sobel > 255) ? 255 : sobel;
      vst1_u16(&img_sobel_out.data[IMG_WIDTH*i + j], sobel);
    }
  }
}
