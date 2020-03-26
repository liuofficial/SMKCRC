/*
Copyright 2011, Ming-Yu Liu

All Rights Reserved 

Permission to use, copy, modify, and distribute this software and 
its documentation for any non-commercial purpose is hereby granted 
without fee, provided that the above copyright notice appear in 
all copies and that both that copyright notice and this permission 
notice appear in supporting documentation, and that the name of 
the author not be used in advertising or publicity pertaining to 
distribution of the software without specific, written prior 
permission. 

THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, 
INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR 
ANY PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR 
ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES 
WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN 
AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE. 
*/

#ifndef _image_h_
#define _image_h_

#include <cmath>
#include <cstring>
#include <assert.h>
typedef unsigned char uchar;

/* use imRef to access Image data. */
#define imRef(im, x, y) (im->access[y][x])
  
/* use imPtr to get pointer to Image data. */
#define imPtr(im, x, y) &(im->access[y][x])

class RGBMap
{
public:
	RGBMap(uchar r,uchar g,uchar b): r_(r),g_(g),b_(b) {};
	RGBMap() {};
	uchar r_, g_, b_;
	inline RGBMap& operator=(const RGBMap &rhs);
	inline double operator-(const RGBMap &other) const;
};

double RGBMap::operator-(const RGBMap &other) const
{
	double diff = 0;
	diff += abs(1.0*r_ - other.r_);
	diff += abs(1.0*g_ - other.g_);
	diff += abs(1.0*b_ - other.b_);
	return diff;
}

RGBMap& RGBMap::operator=(const RGBMap &rhs)
{
	r_ = rhs.r_, g_ = rhs.g_, b_ = rhs.b_;
	return (*this);
}

// modified! --------------------begin-----------------------------------
class HSMap
{
public:
	HSMap(const int *data, int dim);
	HSMap();
	HSMap(const HSMap &map);
	HSMap(int dim);
	~HSMap();
	int* data_;
	int data_dim;
	inline HSMap& operator=(const HSMap &rhs);
	inline double operator-(const HSMap &other) const;
};

double cosine(const HSMap &first, const HSMap &two);
double euclid(const HSMap &first, const HSMap &two);

	
HSMap::HSMap(const int *data, int dim)
{
	data_dim = dim;
	data_ = new int[data_dim];
	for (int i=0; i<data_dim; i++)
		data_[i] = data[i];
}

HSMap::HSMap()
{
	data_ = NULL;
	data_dim = 0;
}

HSMap::HSMap(const HSMap &map)
{
	data_dim = map.data_dim;
	data_ = new int[data_dim];
	for (int i=0; i<data_dim; i++)
		data_[i] = map.data_[i];
}

HSMap::HSMap(int dim)
{
	data_dim = dim;
	data_ = new int[data_dim];
	for (int i=0; i<data_dim; i++)
		data_[i] = 0;
}

HSMap::~HSMap()
{
	delete [] data_;
	data_ = NULL;
	data_dim = 0;
}

double HSMap::operator-(const HSMap &other) const
{
	double diff = 0;
	if (data_dim != other.data_dim) 
	{diff = -1; return diff;}
	
	for (int i=0; i<data_dim; i++)
		diff += abs(1.0*data_[i] - other.data_[i]);
	return diff;
}

double euclid(const HSMap &first, const HSMap &two)
{
	double diff = 0;
	if (first.data_dim != two.data_dim) 
	{diff = -1; return diff;}

	for (int i=0; i<first.data_dim; i++)
		diff += (1.0*first.data_[i] - two.data_[i]) * (1.0*first.data_[i] - two.data_[i]);
	diff = sqrt(diff);
	return diff;
}

double cosine(const HSMap &first, const HSMap &two)
{
	double diff = 0, norm1 = 0, norm2 = 0;
	if (first.data_dim != two.data_dim) 
	{diff = -1; return diff;}

	for (int i=0; i<first.data_dim; i++)
	{
		diff += 1.0*first.data_[i]*two.data_[i];
		norm1 += 1.0*first.data_[i]*first.data_[i];
		norm2 += 1.0*two.data_[i]*two.data_[i];
	}
	diff = acos(diff / sqrt(norm1) / sqrt(norm2));
	return diff;
}

HSMap& HSMap::operator=(const HSMap &rhs)
{
	data_dim = rhs.data_dim;
	data_ = new int[data_dim];
	for (int i=0; i<data_dim; i++)
		data_[i] = rhs.data_[i];
	return (*this);
}
// modified! --------------------end-----------------------------------

template <class T>
class Image // modified not well!
{
	public:

	// constructor
	inline Image();

	/* create an Image */
	inline Image(const int width, const int height, const bool init = true);

	/* delete an Image */
	inline ~Image();

	/* release current image if any */
	inline void Release();

	inline void Resize(const int width,const int height, const bool init = true);
	
	// -----------------------------------new--------------------------------------------------
	inline void Resize(const int width,const int height, const int dim, const bool init = true);

	

	/* init an Image */
	inline void Init(const T &val);

	/* copy an Image */
	inline Image<T> *Copy() const;

	/* get the width of an Image. */
	inline int width() const { return w; }

	/* get the height of an Image. */
	inline int height() const { return h; }

	// returning a reference to the parituclar location.
	inline T& Access(int x,int y) {return access[y][x];};


	/* Image data. */
	T *data;

	/* row pointers. */
	T **access;

	

private:
	int w, h;
};


template <class T>
Image<T>::Image()
{
	w = 0;
	h = 0;
	data = NULL;
	access = NULL;
}


template <class T>
Image<T>::Image(const int width, const int height, const bool init) 
{
	w = width;
	h = height;
	data = new T[w * h];  // allocate space for Image data
	access = new T*[h];   // allocate space for row pointers

	// initialize row pointers
	for (int i = 0; i < h; i++)
		access[i] = data + (i * w);  

	if (init)
		memset(data, 0, w * h * sizeof(T));
}

template <class T>
Image<T>::~Image() 
{
	Release();
}

template <class T>
void Image<T>::Release()
{
	if(data)
		delete [] data;
	if(access)
		delete [] access;

	h = 0;
	w = 0;
}


template <class T>
void Image<T>::Resize(const int width, const int height, const bool init) 
{
	Release();
	w = width;
	h = height;
	data = new T[w * h];  // allocate space for Image data
	access = new T*[h];   // allocate space for row pointers

	// initialize row pointers
	for (int i = 0; i < h; i++)
		access[i] = data + (i * w);  

	if (init)
		memset(data, 0, w * h * sizeof(T));
}

// modified ---------------------------------begin----------------------------------
template <class T>
void Image<T>::Resize(const int width, const int height, const int dim, const bool init) 
{
	Release();
	w = width;
	h = height;
	data = new T[w * h];  // allocate space for Image data
	access = new T*[h];   // allocate space for row pointers

	// initialize row pointers
	for (int i = 0; i < h; i++)
		access[i] = data + (i * w);  

	if (init)
	{
		T zero(dim);
		for (int i=0; i<w*h; i++)
			data[i] = zero;
	}
}
// modified ---------------------------------end----------------------------------

template <class T>
void Image<T>::Init(const T &val) 
{
	T *ptr = imPtr(this, 0, 0);
	T *end = imPtr(this, w-1, h-1);
	while (ptr <= end)
		*ptr++ = val;
}


template <class T>
Image<T> *Image<T>::Copy() const 
{
	Image<T> *im = new Image<T>(w, h, false);
	memcpy(im->data, data, w * h * sizeof(T));
	return im;
}

#endif
  
