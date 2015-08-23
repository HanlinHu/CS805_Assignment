//Name:		Hanlin Hu	
//ID:		200-332-464
//Email:	hu263@uregina.ca


#include <math.h>
#include <stdio.h>
#include <stdlib.h>// for memory allocation function: malloc()

#define ROWS 512
#define COLS 512

//Structure given by MyModel.h --------------------------------------------------------------------
typedef struct {
	float x, y, z;	/* center of the circle */
	float radius;	/* radius of the circle */
	float kd;	/* diffuse reflection coefficient */
} SPHERE;

typedef struct {
	float v[4][3];	/* list of vertices */
	float N[3];	/* normal of the polygon */
	float kd;	/* diffuse reflection coefficient */
} POLY4;

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

	double length = sqrt(dotProduct(p, p));
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

// Ray and Sphere Intersection---------------------------------------------------------------------
float raySphereIntersection(Ray ray, SPHERE sphere, Point3dPtr n1, Point3dPtr interp1, float* kd1)
{
	float t;
	float radius;
	Point3d center;
	Point3dPtr S;
	S = (Point3dPtr)malloc(sizeof(Point3d));
	*kd1 = sphere.kd;

	radius = sphere.radius;
	center.x = sphere.x;
	center.y = sphere.y;
	center.z = sphere.z;
	subVector(&center, ray.a, S);	//S = Sphere Center Translated into Coordinate Frame of Ray Origin

	//Intersection of Sphere and Line     =       Quadratic Function of Distance 
	// A ray is defined by: R(t) = R0 + t * Rd , t > 0 with R0 = [X0, Y0, Z0] and Rd = [Xd, Yd, Zd]
	float A = dotProduct(ray.b, ray.b);		//A = Xd^2 + Yd^2 + Zd^2
	float B = -2.0*dotProduct(S, ray.b);		//B = 2 * (Xd * (X0 - Xc) + Yd * (Y0 - Yc) + Zd * (Z0 - Zc))
	float C = dotProduct(S, S) - radius*radius;	//C = (X0 - Xc)^2 + (Y0 - Yc)^2 + (Z0 - Zc)^2 - Sr^2
	float D = B*B - 4 * A*C;					//Precompute Discriminant

	if (D >= 0.0)
	{
		// if there is a shorter one just use this one, if not use another(longer one).
		int sign = (C < 0.0) ? 1 : -1;
		t = (-B + sign*sqrt(D)) / 2.f; // A should be equal to 1

		// The surface normal
		n1->x = (ray.a->x + ray.b->x*t - center.x) / radius;
		n1->y = (ray.a->y + ray.b->y*t - center.y) / radius;
		n1->z = (ray.a->z + ray.b->z*t - center.z) / radius;

		// The intersection point
		interp1->x = ray.a->x + ray.b->x*t;
		interp1->y = ray.a->y + ray.b->y*t;
		interp1->z = ray.a->z + ray.b->z*t;

		return t;	//The distance
	}
	return 0.0;
	free(S);
}
//Ray polygon intersection-------------------------------------------------------------------------
//Find max amplitude of Surface Normal
int findMax(float a[3])
{
	float temp, max;
	int i, index;
	index = 0;
	max = sqrt(a[0] * a[0]);//max = abs(a[0]);
	for (i = 0; i < 3; i++)
	{
		temp = sqrt(a[i] * a[i]); //temp = abs(a[i]);
		if (temp > max) {
			max = temp;
			index = i;
		}
	}
	return index;
}

//Get the projection plane by compare the x,y,z absolute value of surface normal------------------------------------------------------
void findProjPlane(float a[3], float b[4][3], float d[4][2]) // Surface Normal
{
	int index;

	index = findMax(a);

	if (index == 0)
	{
		d[0][0] = b[0][1];
		d[0][1] = b[0][2];
		d[1][0] = b[1][1];
		d[1][1] = b[1][2];
		d[2][0] = b[2][1];
		d[2][1] = b[2][2];
		d[3][0] = b[3][1];
		d[3][1] = b[3][2];
	}
	else if (index == 1)
	{
		d[0][0] = b[0][0];
		d[0][1] = b[0][2];
		d[1][0] = b[1][0];
		d[1][1] = b[1][2];
		d[2][0] = b[2][0];
		d[2][1] = b[2][2];
		d[3][0] = b[3][0];
		d[3][1] = b[3][2];
	}
	else if (index == 2)
	{
		d[0][0] = b[0][0];
		d[0][1] = b[0][1];
		d[1][0] = b[1][0];
		d[1][1] = b[1][1];
		d[2][0] = b[2][0];
		d[2][1] = b[2][1];
		d[3][0] = b[3][0];
		d[3][1] = b[3][1];
	}
}

//Find projection point----------------------------------------------------------------------------------------------------------
void findProjPoint(float a[3], Point3dPtr p, float dp[2])
{
	int index;
	index = findMax(a);

	if (index == 0)
	{
		dp[0] = p->y;
		dp[1] = p->z;
	}
	else if (index == 1)
	{
		dp[0] = p->x;
		dp[1] = p->z;
	}
	else if (index == 2)
	{
		dp[0] = p->x;
		dp[1] = p->y;
	}
}

//-------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------
// There is a algorithm
// Translate projPoint to origin
// For other projected vertices, 
// the Translation Matrix is
//	|x'|		| 1, 0, -projPoint[0] |		|x|
//	|y'|	=	| 0, 1, -projPoint[1] | *	|y|
//	|1 |		| 0, 0,		1         |		|1|
// Four new points p1,p2,p3,p4 with format (x',y')-------------------------------
// Using each pair of vertices to create a segment: create {vertex1(x1,y1), vertex2(x2,y2)}: from {p1,p2,p3,p4}
// and use 	dy = y2 - y1; dx = x2 - x1;
// M = dy / dx;		//Slope
// line equation : Y = M*X +C
// C = y1 - M*x1;
// Intersect with Y = 0;
// X = (-1)*(C / M);
// init count = 0;
// then, if X >= 0; to make sure we only count positive X axis
// and if (y1 >= 0)&&(y2 < 0) or (y1 < 0)&&(y2 >= 0)
// count += 1;
// else count = count;
// if (count % 2 != 0) : count is odd, the point inside; else outside.
//-------------------------------------------------------------------------------------------------

//The codes of this algorithm  shorted by W.Franklin to seven lines as follow:
//PNPOLY - Point Inclusion in Polygon Test
//by W. Randolph Franklin (WRF)
//http://www.ecse.rpi.edu/~wrf/Research/Short_Notes/pnpoly.html
int pnpoly(int nvert, float *vertx, float *verty, float testx, float testy)
{
	int i, j, c = 0;
	for (i = 0, j = nvert - 1; i < nvert; j = i++) {
		if (((verty[i]>testy) != (verty[j]>testy)) &&
			(testx < (vertx[j] - vertx[i]) * (testy - verty[i]) / (verty[j] - verty[i]) + vertx[i]))
			c = !c;
	}
	return c;
}

//Ray and Polygon Intersection---------------------------------------------------------------------------------------------------
float rayPolygonIntersection(Ray ray, POLY4 poly, Point3dPtr n2, Point3dPtr interp2, float *kd2)
{
	float t;
	float temp1, temp2;
	float projectPlane[4][2];
	float projPoint[2];
	float p0[2];
	float p1[2];
	float p2[2];
	float p3[2];
	float Array[4];
	int c;

	float D;
	Point3d np;

	D = -1 * (poly.N[0] * poly.v[0][0] + poly.N[1] * poly.v[0][1] + poly.N[2] * poly.v[0][2]);  // Put Vertax #1 to calculate value of D
	np = assign_values(poly.N);
	temp1 = dotProduct(&np, ray.a) + D;
	temp2 = dotProduct(&np, ray.b);
	t = -1 * (temp1 / temp2);

	*kd2 = poly.kd;
	// The surface normal------------------------------------------------------------------------------------------------
	n2->x = poly.N[0];
	n2->y = poly.N[1];
	n2->z = poly.N[2];

	// The intersection point--------------------------------------------------------------------------------------------
	interp2->x = ray.a->x + ray.b->x*t;
	interp2->y = ray.a->y + ray.b->y*t;
	interp2->z = ray.a->z + ray.b->z*t;

	// Get Projection of intersection point--------------------------------------------------------------------------------------------
	findProjPoint(poly.N, interp2, projPoint);

	// Get a Projection Plane------------------------------------------------------------------------------------------------------
	findProjPlane(poly.N, poly.v, projectPlane);

	float S0[4];
	float S1[4];
	S0[0] = projectPlane[0][0];
	S0[1] = projectPlane[1][0];
	S0[2] = projectPlane[2][0];
	S0[3] = projectPlane[3][0];

	S1[0] = projectPlane[0][1];
	S1[1] = projectPlane[1][1];
	S1[2] = projectPlane[2][1];
	S1[3] = projectPlane[3][1];

	c = pnpoly(4, S0, S1, projPoint[0], projPoint[1]);
	if ((temp2 > 0.0f) || (temp2 = 0.0f))
		return 0.0f;  	// Ray and Polygon parallel, intersection rejection
	else
	{
		if (c != 0)
			return t;	//The distance
		else
			return 0.0f;
	}
}

//Ray-Object intersection--------------------------------------------------------------------------
float rayObjectIntersection(Ray ray, SPHERE sphere, POLY4 poly, Point3dPtr n, Point3dPtr interp, float * kd)
{
	float t1, t2;		// The two distances;
	float *kd1, *kd2;

	Point3dPtr n1, n2, interp1, interp2;
	n1 = (Point3dPtr)malloc(sizeof(Point3d));
	n2 = (Point3dPtr)malloc(sizeof(Point3d));
	interp1 = (Point3dPtr)malloc(sizeof(Point3d));
	interp2 = (Point3dPtr)malloc(sizeof(Point3d));
	kd1 = (float*)malloc(sizeof(*kd1));
	kd2 = (float*)malloc(sizeof(*kd2));

	t1 = raySphereIntersection(ray, sphere, n1, interp1, kd1);
	t2 = rayPolygonIntersection(ray, poly, n2, interp2, kd2);

	if (t1 == 0.0 && t2 == 0.0)
		return 0.0;
	else if (t2 == 0)
	{
		//the normal = n1;
		n->x = n1->x;
		n->y = n1->y;
		n->z = n1->z;

		//the intersection point = interp1;
		interp->x = interp1->x;
		interp->y = interp1->y;
		interp->z = interp1->z;

		*kd = *kd1;
		return t1;
	}
	else if (t1 == 0.0)
	{
		//the normal = n2;
		n->x = n2->x;
		n->y = n2->y;
		n->z = n2->z;

		//the intersection point = interp2;
		interp->x = interp2->x;
		interp->y = interp2->y;
		interp->z = interp2->z;

		*kd = *kd2;
		return t2;
	}
	else if (t1 < t2)
	{
		//the normal = n1;
		n->x = n1->x;
		n->y = n1->y;
		n->z = n1->z;

		//the intersection point = interp1;
		interp->x = interp1->x;
		interp->y = interp1->y;
		interp->z = interp1->z;

		*kd = *kd1;
		return t1;
	}
	else
	{
		//the normal = n2;
		n->x = n2->x;
		n->y = n2->y;
		n->z = n2->z;

		//the intersection point = interp2;
		interp->x = interp2->x;
		interp->y = interp2->y;
		interp->z = interp2->z;

		*kd = *kd2;
		return t2;
	}
	free(n1);
	free(n2);
	free(interp1);
	free(interp2);
	free(kd1);
	free(kd2);
}

//Shading------------------------------------------------------------------------------------------
int shading(float lrp[3], Point3dPtr n, Point3dPtr p, float * kd, float ip)
{
	Point3dPtr l, ln;
	float temp = 0.0f;
	unsigned char C = 0;

	l = (Point3dPtr)malloc(sizeof(Point3d));
	ln = (Point3dPtr)malloc(sizeof(Point3d));

	l->x = lrp[0] - p->x;
	l->y = lrp[1] - p->y;
	l->z = lrp[2] - p->z;
	normalize(l, ln);
	temp = dotProduct(n, ln);

	// face side is valid; back side is invalid
	if (temp < 0)
		temp = 0;


	//float s = (*kd) * ip * temp;
	C = (unsigned char)(int)ip * (*kd) * temp;
	return C;
}

// Ray tracing
int rayTracing(Ray ray, SPHERE sphere, POLY4 poly, float lrp[3], float ip)
{
	float P;
	Point3dPtr n;
	Point3dPtr interp;

	int C;
	float * kd = (float*)malloc(sizeof(*kd));

	n = (Point3dPtr)malloc(sizeof(Point3d));
	interp = (Point3dPtr)malloc(sizeof(Point3d));

	P = rayObjectIntersection(ray, sphere, poly, n, interp, kd);

	if (P != 0.0)
	{
		C = shading(lrp, n, interp, kd, ip);
		return C;
	}
	else
		return 0;
	free(n);
	free(interp);
	free(kd);
}

//Main---------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{

	//Variables from MyModel.h---------------------------------------------------------------------

	/* create a spherical object */
	SPHERE obj1 = { 1.0, 1.0, 1.0,	/* center of the circle */
		1.0,		/* radius of the circle */
		0.75 };		/* diffuse reflection coefficient */

	/* create a polygon object */
	POLY4 obj2 = { 0.0, 0.0, 0.0,	/* v0 */
		0.0, 0.0, 2.0,	/* v1 */
		2.0, 0.0, 2.0,	/* v2 */
		2.0, 0.0, 0.0,	/* v3 */
		0.0, 1.0, 0.0,	/* normal of the polygon */
		0.8 };		/* diffuse reflection coefficient */

	unsigned char img[ROWS][COLS];
	float xmin = 0.0175;
	float ymin = -0.0175;
	float xmax = -0.0175;
	float ymax = 0.0175;

	float focal = 0.05;	/* focal length simulating 50 mm lens */

	/* definition of the camera parameters */
	float VRP[3] = { 1.0, 2.0, 3.5 };
	float VPN[3] = { 0.0, -1.0, -2.5 };
	float VUP[3] = { 0.0, 1.0, 0.0 };

	/* definition of light source */
	float LRP[3] = { -10.0, 10.0, 2.0 };	/* light position */
	float Ip = 200.0;	/* intensity of the point light source */

	//---------------------------------------------------------------------------------------------
	int i, j;
	int c;
	Xform3d TIN1, RT1;
	Xform3d Mcw;
	Point3d Pvpn, Pvup, Pvrp;

	Pvpn = assign_values(VPN);
	Pvup = assign_values(VUP);
	Pvrp = assign_values(VRP);

	translationInverse(VRP, TIN1);
	rotationTranspose(&Pvpn, &Pvup, RT1);

	multXforms(TIN1, RT1, Mcw);

	// initiate buffer
	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			img[i][j] = 0;
		}
	}

	for (i = 0; i < ROWS; i++)
	{
		for (j = 0; j < COLS; j++)
		{
			Ray V;
			V.a = (Point3dPtr)malloc(sizeof(Point3d));
			V.b = (Point3dPtr)malloc(sizeof(Point3d));
			rayConstruction(i, j, focal, xmin, xmax, ymin, ymax, Mcw, &Pvrp, &V);// construct Ray V
			c = rayTracing(V, obj1, obj2, LRP, Ip);
			img[i][j] = c;

			free(V.a);
			free(V.b);
		}
	}

	// function output the final image to binary, then change to tiff or other format by using third party tools, such as Photoshop etc.
	FILE * fp;
	fp = fopen("Ray.raw", "wb");
	fwrite(img, sizeof(unsigned char), sizeof(img), fp);
	fclose(fp);

	return 0;
}
