//name: Hanlin Hu
//ID:	200-332-464
//Email:hu263@uregina.ca

//---------------------------------------------

#include <math.h>
#include <stdio.h>
#include <stdlib.h>


/*Macros*/
#define SLCS 128
#define ROWS 128
#define COLS 128
#define IMG_ROWS 512
#define IMG_COLS 512

//---------------------------------------------
//Type definitions---------------------------------------------------------------------------------
typedef struct Point3d
{
	float x, y, z;

}Point3d, *Point3dPtr;

typedef struct Ray
{
	Point3dPtr a, b;
} Ray;
//Assign value from an Array to a Point3d----------------------------------------------------------
Point3d assign_values(float a[3])
{
	Point3d point;
	point.x = a[0];
	point.y = a[1];
	point.z = a[2];

	return point;
}
//Math---------------------------------------------------------------------------------------------
void addVector(Point3dPtr p1, Point3dPtr p2, Point3dPtr result)
{
	result->x = p1->x + p2->x;
	result->y = p1->y + p2->y;
	result->z = p1->z + p2->z;
}

void subVector(Point3dPtr p1, Point3dPtr p2, Point3dPtr result1)
{
	result1->x = p1->x - p2->x;
	result1->y = p1->y - p2->y;
	result1->z = p1->z - p2->z;
}

void mulVectorScalar(Point3dPtr p1, float c, Point3dPtr result2)
{
	result2->x = c*p1->x;
	result2->y = c*p1->y;
	result2->z = c*p1->z;
}

float dotProduct(Point3dPtr p1, Point3dPtr p2)
{
	return p1->x*p2->x + p1->y*p2->y + p1->z*p2->z;
}

void crossProduct(Point3dPtr p1, Point3dPtr p2, Point3dPtr result3)
{
	result3->x = p1->y * p2->z - p1->z * p2->y;
	result3->y = p1->z * p2->x - p1->x * p2->z;
	result3->z = p1->x * p2->y - p1->y * p2->x;
}

void normalize(Point3dPtr p, Point3dPtr result4)
{

	float length = sqrt(dotProduct(p, p));
	if (length != 0.0)
	{
		p->x /= length;
		p->y /= length;
		p->z /= length;
	}
	result4->x = p->x;
	result4->y = p->y;
	result4->z = p->z;
}

//Compute Mcw--------------------------------------------------------------------------------------
typedef float Xform3d[4][4];

void rotationTranspose(Point3dPtr p2, Point3dPtr p3, Xform3d rotTMatrix)
{
	Point3d	tempU, u, v, n;
	normalize(p2, &n);//calculate n

	crossProduct(p3, p2, &tempU);	//calculate crossProduct as numerator 
	normalize(&tempU, &u);	//calculate normalization as denominator

	crossProduct(&n, &u, &v);

	rotTMatrix[0][0] = u.x;
	rotTMatrix[0][1] = v.x;
	rotTMatrix[0][2] = n.x;
	rotTMatrix[0][3] = 0.0;

	rotTMatrix[1][0] = u.y;
	rotTMatrix[1][1] = v.y;
	rotTMatrix[1][2] = n.y;
	rotTMatrix[1][3] = 0.0;

	rotTMatrix[2][0] = u.z;
	rotTMatrix[2][1] = v.z;
	rotTMatrix[2][2] = n.z;
	rotTMatrix[2][3] = 0.0;

	rotTMatrix[3][0] = 0.0;
	rotTMatrix[3][1] = 0.0;
	rotTMatrix[3][2] = 0.0;
	rotTMatrix[3][3] = 1.0;

}

void translationInverse(float vrp[3], Xform3d transInMatrix)
{

	transInMatrix[0][0] = 1.0;
	transInMatrix[0][1] = 0.0;
	transInMatrix[0][2] = 0.0;
	transInMatrix[0][3] = vrp[0];

	transInMatrix[1][0] = 0.0;
	transInMatrix[1][1] = 1.0;
	transInMatrix[1][2] = 0.0;
	transInMatrix[1][3] = vrp[1];

	transInMatrix[2][0] = 0.0;
	transInMatrix[2][1] = 0.0;
	transInMatrix[2][2] = 1.0;
	transInMatrix[2][3] = vrp[2];

	transInMatrix[3][0] = 0.0;
	transInMatrix[3][1] = 0.0;
	transInMatrix[3][2] = 0.0;
	transInMatrix[3][3] = 1.0;
}

void copy3dXform(Xform3d dst, Xform3d src)
{
	register int i, j;
	for (i = 0; i < 4; i++)
	for (j = 0; j < 4; j++)
		dst[i][j] = src[i][j];
}

void multXforms(Xform3d xform1, Xform3d xform2, Xform3d resultxform)
{
	Xform3d result;
	//-------------------------------------row 0
	result[0][0] = xform1[0][0] * xform2[0][0] +
		xform1[0][1] * xform2[1][0] +
		xform1[0][2] * xform2[2][0] +
		xform1[0][3] * xform2[3][0];
	result[0][1] = xform1[0][0] * xform2[0][1] +
		xform1[0][1] * xform2[1][1] +
		xform1[0][2] * xform2[2][1] +
		xform1[0][3] * xform2[3][1];
	result[0][2] = xform1[0][0] * xform2[0][2] +
		xform1[0][1] * xform2[1][2] +
		xform1[0][2] * xform2[2][2] +
		xform1[0][3] * xform2[3][2];
	result[0][3] = xform1[0][0] * xform2[0][3] +
		xform1[0][1] * xform2[1][3] +
		xform1[0][2] * xform2[2][3] +
		xform1[0][3] * xform2[3][3];
	//-------------------------------------row 1
	result[1][0] = xform1[1][0] * xform2[0][0] +
		xform1[1][1] * xform2[1][0] +
		xform1[1][2] * xform2[2][0] +
		xform1[1][3] * xform2[3][0];
	result[1][1] = xform1[1][0] * xform2[0][1] +
		xform1[1][1] * xform2[1][1] +
		xform1[1][2] * xform2[2][1] +
		xform1[1][3] * xform2[3][1];
	result[1][2] = xform1[1][0] * xform2[0][2] +
		xform1[1][1] * xform2[1][2] +
		xform1[1][2] * xform2[2][2] +
		xform1[1][3] * xform2[3][2];
	result[1][3] = xform1[1][0] * xform2[0][3] +
		xform1[1][1] * xform2[1][3] +
		xform1[1][2] * xform2[2][3] +
		xform1[1][3] * xform2[3][3];
	//-------------------------------------row 2
	result[2][0] = xform1[2][0] * xform2[0][0] +
		xform1[2][1] * xform2[1][0] +
		xform1[2][2] * xform2[2][0] +
		xform1[2][3] * xform2[3][0];
	result[2][1] = xform1[2][0] * xform2[0][1] +
		xform1[2][1] * xform2[1][1] +
		xform1[2][2] * xform2[2][1] +
		xform1[2][3] * xform2[3][1];
	result[2][2] = xform1[2][0] * xform2[0][2] +
		xform1[2][1] * xform2[1][2] +
		xform1[2][2] * xform2[2][2] +
		xform1[2][3] * xform2[3][2];
	result[2][3] = xform1[2][0] * xform2[0][3] +
		xform1[2][1] * xform2[1][3] +
		xform1[2][2] * xform2[2][3] +
		xform1[2][3] * xform2[3][3];
	//-------------------------------------row 3
	result[3][0] = xform1[3][0] * xform2[0][0] +
		xform1[3][1] * xform2[1][0] +
		xform1[3][2] * xform2[2][0] +
		xform1[3][3] * xform2[3][0];
	result[3][1] = xform1[3][0] * xform2[0][1] +
		xform1[3][1] * xform2[1][1] +
		xform1[3][2] * xform2[2][1] +
		xform1[3][3] * xform2[3][1];
	result[3][2] = xform1[3][0] * xform2[0][2] +
		xform1[3][1] * xform2[1][2] +
		xform1[3][2] * xform2[2][2] +
		xform1[3][3] * xform2[3][2];
	result[3][3] = xform1[3][0] * xform2[0][3] +
		xform1[3][1] * xform2[1][3] +
		xform1[3][2] * xform2[2][3] +
		xform1[3][3] * xform2[3][3];

	copy3dXform(resultxform, result);
}

//Ray Construction---------------------------------------------------------------------------------
float map(float max, float min, int a, int CorR)
{
	float temp1, temp2, result;
	temp1 = min - max;
	temp2 = (temp1 * a) / (CorR - 1); //(map from (-0.0175~0.0175) to (0~511)
	result = max + temp2;
	return result;
}

void rayConstruction(int i, int j, float focal, float xmin, float xmax, float ymin, float ymax, float Mcw[4][4], Point3dPtr vrp, Ray *ray)
{

	Point3dPtr  p0, p1, v0, vn0;
	float xc, yc, f;

	p0 = (Point3dPtr)malloc(sizeof(Point3d));
	p1 = (Point3dPtr)malloc(sizeof(Point3d));
	v0 = (Point3dPtr)malloc(sizeof(Point3d));
	vn0 = (Point3dPtr)malloc(sizeof(Point3d));

	xc = map(xmax, xmin, j, COLS);
	yc = map(ymax, ymin, i, ROWS);
	f = focal;

	// vrp is already a 3d Point in the world coordinates , so it doesn't need transform.
	p0->x = vrp->x;
	p0->y = vrp->y;
	p0->z = vrp->z;

	// the points in the film ---- pixels need to be transformed by Mcw.
	//| x'|		|			  |		| x |
	//| y'| =	|  Mcw[4][4]  | *	| y |
	//| z'|		|			  |		| z |
	//| 1 |		|			  |     | 1 |	
	p1->x = Mcw[0][0] * xc + Mcw[0][1] * yc + Mcw[0][2] * f + Mcw[0][3];
	p1->y = Mcw[1][0] * xc + Mcw[1][1] * yc + Mcw[1][2] * f + Mcw[1][3];
	p1->z = Mcw[2][0] * xc + Mcw[2][1] * yc + Mcw[2][2] * f + Mcw[2][3];

	// direction
	v0->x = p1->x - p0->x;
	v0->y = p1->y - p0->y;
	v0->z = p1->z - p0->z;
	normalize(v0, vn0);

	// finally the ray expression : R(t) = R0 + t * Rd , t > 0. Here is ray->a + t * (ray.b) ;
	ray->a->x = p0->x;
	ray->a->y = p0->y;
	ray->a->z = p0->z;

	ray->b->x = vn0->x;
	ray->b->y = vn0->y;
	ray->b->z = vn0->z;
}
//-------------------------------------------------------------------------------------------------
// Tri-linear interpolation function 
float triInter(unsigned char array[SLCS][ROWS][COLS], Point3dPtr curP)
{
	int X = floor(curP->x);
	int Y = floor(curP->y);
	int Z = floor(curP->z);

	float x = curP->x - X;
	float y = curP->y - Y;
	float z = curP->z - Z;

	return 	array[X][Y][Z] * (1 - x)*(1 - y)*(1 - z)
		+ array[X][Y][Z + 1] * (1 - x)*(1 - y)*(z)
		+array[X][Y + 1][Z] * (1 - x)*(y)*(1 - z)
		+ array[X][Y + 1][Z + 1] * (1 - x)*(y)*(z)
		+array[X + 1][Y][Z] * (x)*(1 - y)*(1 - z)
		+ array[X + 1][Y][Z + 1] * (x)*(1 - y)*(z)
		+array[X + 1][Y + 1][Z] * (x)*(y)*(1 - z)
		+ array[X + 1][Y + 1][Z + 1] * (x)*(y)*(z);
}

// Compute the shading volume

void computeShadingVolume(float lrp[3], unsigned char ct[SLCS][ROWS][COLS], float ip, unsigned char shading[SLCS][ROWS][COLS])
{
	int i, j, k;
	Point3dPtr l, ln, n, nn;
	float temp = 0.0f;
	float lengthOfn;
	float threshold = 0.01f;

	l = (Point3dPtr)malloc(sizeof(Point3d));
	ln = (Point3dPtr)malloc(sizeof(Point3d));
	n = (Point3dPtr)malloc(sizeof(Point3d));
	nn = (Point3dPtr)malloc(sizeof(Point3d));

	for (k = 0; k<SLCS; k++)
	for (j = 0; j<ROWS; j++)
	for (i = 0; i<COLS; i++)
	{
		l->x = lrp[0] - i;
		l->y = lrp[1] - j;
		l->z = lrp[2] - k;
		normalize(l, ln);

		n->x = ct[k][j][i+1] - ct[k][j][i];
		n->y = ct[k][j+1][i] - ct[k][j][i];
		n->z = ct[k+1][j][i] - ct[k][j][i];
		normalize(n, nn);
		temp = dotProduct(nn, ln);

		// face side is valid; back side is invalid
		if (temp < 0)
			temp = 0;

		lengthOfn = sqrt(n->x*n->x + n->y*n->y + n->z*n->z);

		if (lengthOfn < threshold)
			shading[k][j][i] = 0;
		else
			//unsigned char COLOR[k][j][i] = Ip*kd* Max(N*L, 0);
			shading[k][j][i] = (unsigned char)(int)ip * temp;
	}
}

unsigned char volumeRayTracing(Ray ray, float ts[2], unsigned char ct[SLCS][ROWS][COLS], unsigned char shading[SLCS][ROWS][COLS])
{
	/*Assume front to back integration*/
	//float Dt = 20.0; 	//the interval for sampling along the ray
	// Why 20? Because focal length is 0.05. We have 20 times length of unit if we do not normalize it. 

	float Dt = 1.0;
	float C = 0.0;		//for accumulating the shading value
	float T = 1.0;		//for accumulating the transparency
	float Alpha, Ci;
	float t0, t1, temp, t;
	unsigned char img = 0;

	t0 = ts[0];
	t1 = ts[1];

	// Swap function
	if (t0 > t1)
	{
		temp = t0;
		t0 = t1;
		t1 = temp;
	}

	/*
	Marching through the CT volume from t0 to t1 by step size Dt
	*/
	for (t = t0; t <= t1; t += Dt)
	{
		Point3dPtr currentPoint;
		currentPoint = (Point3dPtr)malloc(sizeof(Point3d));

		currentPoint->x = ray.a->x + ray.b->x*t;
		currentPoint->y = ray.a->y + ray.b->y*t;
		currentPoint->z = ray.a->z + ray.b->z*t;

		/*
		Alpha must be in the range [0, 1];
		The density value in CT[][][] is in the range [0, 255];
		*/
		Alpha = triInter(ct, currentPoint) / 255.f;
		Ci = triInter(shading, currentPoint);
		C += Ci*Alpha;
	}
	img = (unsigned char)(int)C;
	return(img);
}

// Ray-CT_Box intersection
/*
Because CT_Box is not as the same as the common box.
The Boundaries are  defined when we read the CT data to CT[i][0][0],
which means the CT cube is formed by slices from (x = 0) side to (x = 127) side.
The binary data file containing 128*128*128 unsigned char type.
*/
int rayCTBoxIntersection(Ray *ray, float ts[2])
{
	int n;
	float x0, x1, x2, x3, y0, y1, y2, y3, z0, z1, z2, z3;
	float t0, t1, t2, t3, t4, t5;
	n = 0;

	// Side x = 0; 
	t0 = -1 * ray->a->x / ray->b->x; //x(t) = P0[0]+ V[0] * t = 0.0;
	y0 = ray->a->y + t0 * ray->b->y;
	z0 = ray->a->z + t0 * ray->b->z;
	// Side x = 127;
	t1 = (127 - ray->a->x) / ray->b->x;// x(t) = P0[0] + V[0]*t = 127;
	y1 = ray->a->y + t1 * ray->b->y;
	z1 = ray->a->z + t1 * ray->b->z;

	// Side y = 0;
	t2 = -1 * ray->a->y / ray->b->y; //y(t) = P0[0]+ V[0] * t = 0.0;
	x0 = ray->a->x + t2 * ray->b->x;
	z2 = ray->a->z + t2 * ray->b->z;
	// Side y = 127;
	t3 = (127 - ray->a->y) / ray->b->y; //y(t) = P0[0]+ V[0] * t = 127;
	x1 = ray->a->x + t3 * ray->b->x;
	z3 = ray->a->z + t3 * ray->b->z;

	// Side z = 0;
	t4 = -1 * ray->a->z / ray->b->z; //z(t) = P0[0]+ V[0] * t = 0.0;
	x2 = ray->a->x + t4 * ray->b->x;
	y2 = ray->a->y + t4 * ray->b->y;
	// Side z =127;	
	t5 = (127 - ray->a->z) / ray->b->z; //z(t) = P0[0]+ V[0] * t = 127;
	x3 = ray->a->x + t5 * ray->b->x;
	y3 = ray->a->y + t5 * ray->b->y;

	if (y0>0 && y0<ROWS && z0>0 && z0<SLCS)
	{
		//save t value into ts[];
		ts[n] = t0;
		n += 1; //update n;
	}
	if (y1>0 && y1<ROWS && z1>0 && z1<SLCS)
	{
		//save t value into ts[];
		ts[n] = t1;
		n += 1; //update n;
	}
	if (x0>0 && x0<COLS && z2>0 && z2<SLCS)
	{
		//save t value into ts[];
		ts[n] = t2;
		n += 1; //update n;
	}
	if (x1>0 && x1<COLS && z3>0 && z3<SLCS)
	{
		//save t value into ts[];
		ts[n] = t3;
		n += 1; //update n;
	}
	if (y2>0 && y2<ROWS && x2>0 && x2<COLS)
	{
		//save t value into ts[];
		ts[n] = t4;
		n += 1; //update n;
	}
	if (y3>0 && y3<ROWS && x3>0 && x3<COLS)
	{
		//save t value into ts[];
		ts[n] = t5;
		n += 1; //update n;
	}
	return n;
}


//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
	//initialize the global data-------------------------
	unsigned char	CT[SLCS][ROWS][COLS]; /* a 3D array for CT data */
	unsigned char	SHADING[SLCS][ROWS][COLS]; /* a 3D array for shading values */
	unsigned char	out_img[IMG_ROWS][IMG_COLS];

	/* Camera parameters */
	float VRP[3] = { 128.0, 64.0, 250.0 };
	float VPN[3] = { -64.0, 0.0, -186.0 };
	float VUP[3] = { 0.0, 1.0, 0.0 };

	/* Image Plane Sizes */
	float focal = 0.05;	/* 50 mm lens */
	float xmin = -0.0175;	/* 35 mm "film" */
	float ymin = -0.0175;
	float xmax = 0.0175;
	float ymax = 0.0175;

	/* Light direction (unit length vector) */
	float Light[3] = { 0.577, -0.577, -0.577 };

	/* Light Intensity */
	float Ip = 255.0;
	//---------------------------------------------------

	FILE *infid, *outfid;	// input and output file id's
	int n;
	int d, e, f;

	// Load the CT data into the array
	if ((infid = fopen("smallHead.den", "rb")) == NULL)
	{
		printf("Open CT DATA File Error.\n");
		exit(1);
	}
	for (f = 0; f<SLCS; f++)
	{
		n = fread(&CT[f][0][0], sizeof(char), ROWS*COLS, infid);
		if (n<ROWS*COLS*sizeof(char))
		{
			printf("Read CT data slice %d error.\n", f);
			exit(1);
		}
	}
	//compute the shading volume
	computeShadingVolume(Light, CT, Ip, SHADING);
	//-------------------------------------

	/*
	The Main Ray-Tracing Volume Rendering Part
	*/
	int inter; //"inter" is the number of intersection found; 
	float ts[2];// for storing the intersection points t0 and t1;
	Xform3d TIN1, RT1;
	Xform3d Mcw;
	Point3d Pvpn, Pvup, Pvrp;

	Pvpn = assign_values(VPN);
	Pvup = assign_values(VUP);
	Pvrp = assign_values(VRP);

	translationInverse(VRP, TIN1);
	rotationTranspose(&Pvpn, &Pvup, RT1);

	multXforms(TIN1, RT1, Mcw);

	for (d = 0; d<IMG_ROWS; d++)
	for (e = 0; e<IMG_COLS; e++)
	{
		//Construct a ray V, started from the CenterOfProjection and passing through the pixel(i,j);	
		Ray V;
		V.a = (Point3dPtr)malloc(sizeof(Point3d));
		V.b = (Point3dPtr)malloc(sizeof(Point3d));
		rayConstruction(d, e, focal, xmin, xmax, ymin, ymax, Mcw, &Pvrp, &V);// construct Ray V

		// Find how many intersections with CT box
		inter = rayCTBoxIntersection(&V, ts);
		if (inter == 2)
			out_img[d][e] = volumeRayTracing(V, ts, CT, SHADING);

		free(V.a);
		free(V.b);
	}
	/*Save the output image*/

	outfid = fopen("outImage.raw", "wb");
	n = fwrite(out_img, sizeof(char), IMG_ROWS*IMG_COLS, outfid);
	if (n < IMG_ROWS*IMG_COLS*sizeof(char))
	{
		printf("Write output image error.\n");
		exit(1);
	}
	fclose(infid);
	fclose(outfid);
	exit(0);

	return 0;
}
