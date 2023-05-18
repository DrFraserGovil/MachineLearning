#pragma once
#include <cmath>
double ReLUActivation(double x)
{
	if (x < 0)
	{
		return 0;
	}
	return x;
}
double ReLUDerivative(double x)
{
	if (x < 0)
	{
		return 0;
	}
	return 1;
}
double LogisticActivation(double x)
{
	// x = std::max(-50.0,std::min(50.0,x));
	return 1.0/(1.0 + exp(-x));
}
double LogisticDerivative(double x)
{
	// x = std::max(-50.0,std::min(50.0,x));
	double ex = exp(-x);
	double v = (1.0 + ex);
	return ex/(v*v);
}
double q = 0.1;
double LinearActivation(double x)
{
	// x = std::max(-50.0,std::min(50.0,x));
	return x*q;
}
double LinearDerivative(double x)
{
	return q;
}
double m = 0.35;
double SinusoidActivation(double x)
{
	// x = std::max(-50.0,std::min(50.0,x));
	return sin(m*x);
}
double SinusoidDerivative(double x)
{
	return m*cos(m*x);
}