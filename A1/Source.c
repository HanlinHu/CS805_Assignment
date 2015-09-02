//Name:		Hanlin Hu	
//Email:	hu263@cs.uregina.ca

#include <stdio.h>
#include <stdlib.h>
#include <math.h>



typedef struct Point3d  /* the 3D homogeneous point */
{
    double x, y, z, w;
}Point3d, *Point3dPtr;


Point3d assign_values(double a[3])
{
    Point3d point;
	point.x = a[0];
	point.y = a[1];
	point.z = a[2];
	point.w = 1.0;

	return point;
}


double dotProduct(Point3dPtr p1, Point3dPtr p2)
{
	return p1->x*p2->x + p1->y*p2->y + p1->z*p2->z;
}

void crossProduct(Point3dPtr p1, Point3dPtr p2, Point3dPtr result)
{
    result->x = p1->y * p2->z - p1->z * p2->y; //23-32;
	result->y = p1->z * p2->x - p1->x * p2->z; //31-13;
	result->z = p1->x * p2->y - p1->y * p2->x; //12-21;
}

void normalize(Point3dPtr p, Point3dPtr result)
{    
	
	double length = sqrt(dotProduct(p, p));
	if (length != 0.0)
	{
		p->x /= length;
		p->y /= length;
		p->z /= length;
	}
	result->x = p->x;
	result->y = p->y;
	result->z = p->z;
}

int main(int argc, char* argv[])
{	
	Point3d p2, p3, tempU, u, v, n;
	double V1[3] = {0.0,0.0,3.0};
	double V2[3] = {0.0,2.0,5.0};

	p2 = assign_values(V1);
    p3 = assign_values(V2);
    
    
	normalize(&p2,&n);//calculate n
    
	crossProduct(&p3,&p2,&tempU);//calculate crossProduct as numerator
    normalize(&tempU, &u);//calculate normalization as u
    
    crossProduct(&n, &u, &v);//calculate v
    
    printf("ux=%f, uy=%f, uz=%f\n",u.x,u.y,u.z);
    printf("vx=%f, vy=%f, vz=%f\n",v.x,v.y,v.z);
    printf("nx=%f, ny=%f, nz=%f\n",n.x,n.y,n.z);
	
	return 0;
}

//Result--------------
/*
ux=1.000000, uy=0.000000, uz=0.000000
vx=0.000000, vy=1.000000, vz=0.000000
nx=0.000000, ny=0.000000, nz=1.000000
*/
