#include "libcolordescriptors.h"

void Extraction(char *img_path, char *fv_path)
{
  CImage *cimg=NULL;
  Image *mask=NULL;
  Histogram *bic=NULL;

  cimg = ReadCImage(img_path);
  mask = CreateImage(cimg->C[0]->ncols, cimg->C[0]->nrows);

  bic = BIC(cimg, mask);

  WriteFileHistogram(bic,fv_path);
  DestroyHistogram(&bic);
  DestroyImage(&mask);
  DestroyCImage(&cimg);

}

void* LoadFV(char* fv_path) {
    return (void*) ReadFileHistogram(fv_path);
}

double Distance(void *fv1, void *fv2)
{
  Histogram *bic1=NULL;
  Histogram *bic2=NULL;
  double distance;

  bic1 = (Histogram *) fv1;
  bic2 = (Histogram *) fv2;

  distance = (double) L1Distance(bic1, bic2);

//   DestroyHistogram(&bic1);
//   DestroyHistogram(&bic2);

  return distance;
}
