#include "libcolordescriptors.h"

void Extraction(char *img_path, char *fv_path)
{
  CImage *cimg=NULL;
  Image *mask=NULL;
  FeatureVector1DInt *bic=NULL;

  cimg = ReadCImage(img_path);
  mask = CreateImage(cimg->C[0]->ncols, cimg->C[0]->nrows);

  bic = BIC(cimg, mask);

  WriteFileFeatureVector1DInt(bic,fv_path);
  DestroyFeatureVector1DInt(&bic);
  DestroyCImage(&cimg);
  DestroyImage(&mask);

}

double Distance(char *fv1_path, char *fv2_path)
{
  FeatureVector1DInt *bic1=NULL;
  FeatureVector1DInt *bic2=NULL;
  double distance;

  bic1 = ReadFileFeatureVector1DInt(fv1_path);
  bic2 = ReadFileFeatureVector1DInt(fv2_path);

  distance = (double) EuclideanDistanceInt(bic1, bic2);

  DestroyFeatureVector1DInt(&bic1);
  DestroyFeatureVector1DInt(&bic2);

  return distance;
}
