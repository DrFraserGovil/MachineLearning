#include "JSL/JSL.h" //JSL is a custom library that I use which contains both the command line argument processing and the plotting library
#include "MLP.h"
#include "activationFunctions.h"
#include <random>
int mode = 0;
int activation = 0;


struct LearningData
{
	std::vector<std::vector<double>> TrainingFeatures;
	std::vector<std::vector<double>> TrainingTruths;
	std::vector<std::vector<double>> ValidationFeatures;
	std::vector<std::vector<double>> ValidationTruths;	

	LearningData(std::vector<std::vector<double>> features, std::vector<std::vector<double>> truths, double trainingFrac)
	{
		std::vector<int> shuffler(truths.size());
	std::iota(shuffler.begin(),shuffler.end(),0);
	
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(shuffler.begin(),shuffler.end(),g);

	for (int i =0; i < shuffler.size(); ++i)
	{
		if (i < trainingFrac*shuffler.size())
		{
			TrainingTruths.push_back(truths[shuffler[i]]);
			TrainingFeatures.push_back(features[shuffler[i]]);
		}
		else
		{
			ValidationFeatures.push_back(features[shuffler[i]]);
			ValidationTruths.push_back(truths[shuffler[i]]);
		}
	}
	}
};

//generates the dataset
LearningData SynthesiseData(int n)
{
	std::random_device rd;
	std::mt19937 g(rd());
	std::vector<std::vector<double>> feature;
	std::vector<std::vector<double>> vals;
	for (int i = 0; i < n; ++i)
	{
		double x = 1.2*((double)g()/g.max() - 0.5);
		double y = 1.2*((double)g()/g.max() - 0.5);
		feature.push_back({x,y});
		

		bool condition;
		if (mode == 0)
		{
			bool inC1 = sqrt(x*x + y*y) < 0.2;
			bool inC2 = sqrt(x*x + y*y) < 0.55;
			condition = !inC1 && inC2;
		}
		if (mode == 1)
		{
			bool xLow = x<0;
			bool yLow = y<0;
			condition =  (xLow && !yLow) || (!xLow && yLow);
		}
		if (mode == 2)
		{
			bool inC1 = sqrt(x*x + y*y) < 0.2;
			bool inC2 = sqrt(x*x + y*y) < 0.55;
			bool cutout = abs(y)<0.1;
			condition = !inC1 && inC2 && !cutout;
		}
		if (mode == 3)
		{
			bool below1 = (y < 0.4 * x  + 0.2*sin(12*x)+0.3);
			bool above1 = (y > 0.35 * x  + 0.1*sin(40*x)-0.3);
			bool innerCirc = pow(x/0.4,2) + pow(y/0.2,2) < 1;
			condition = below1 && above1 && !innerCirc;
		}

		if (condition)
		{
			vals.push_back({0.0});
		}
		else
		{
			vals.push_back({1.0});
		}
	}
	return LearningData(feature,vals,0.7);
}

void DisplayProgress(MLP & mlp,std::vector<std::vector<double>> dataX, std::vector<std::vector<double>> dataY)
{
	std::vector<double> x = JSL::Vector::linspace(-0.6,0.6,300);
	std::vector<double> y = JSL::Vector::linspace(-0.6,0.6,300);
	std::vector<std::vector<double>> grid(x.size(),std::vector(y.size(),0.0));
	for (int i = 0; i < x.size(); ++i)
	{
		for (int j = 0; j < y.size(); ++j)
		{
			std::vector<double> input = {x[i],y[j]};
			auto out = mlp.Process(input);
			grid[j][i] = out[0];
		}
	}

	//this is a custom plotting interface I use via JSL, 
	JSL::gnuplot gp;
	gp.WindowSize(700,700);
	gp.Map(x,y,grid);
	gp.Scatter(dataX[0],dataY[0],JSL::LineProperties::ScatterType(JSL::Dot),JSL::LineProperties::PenSize(3));
	gp.Scatter(dataX[1],dataY[1],JSL::LineProperties::ScatterType(JSL::Dot),JSL::LineProperties::PenSize(3));
	gp.SetPalette("gray");
	gp.SetXRange(-0.6,0.6);
	gp.SetYRange(-0.6,0.6);

	gp.Show();
}



int layerLoops = 4;
void MLP_Demonstrate(int N)
{
	//generate and separate the data
	LearningData Data = SynthesiseData(N);
	std::vector<std::vector<double>> xs(2);
	std::vector<std::vector<double>> ys(2);
	for (int i =0; i < Data.ValidationFeatures.size(); ++i)
	{
		int v = Data.ValidationTruths[i][0];
		xs[v].push_back(Data.ValidationFeatures[i][0]);
		ys[v].push_back(Data.ValidationFeatures[i][1]);
	}

	

	//choose the different activation functions
	double (*func)(double x) = LogisticActivation;
	double (*derivative)(double x) = LogisticDerivative;
	switch (activation)
	{
		case 0:
		{
			func = LogisticActivation;
			derivative = LogisticDerivative;
			break;
		}
		case 1:
		{
			func = LogisticActivation;
			derivative= LogisticDerivative;
			break;
		}
		case 2:
		{
			func = LinearActivation;
			derivative = LinearDerivative;
			break;
		}
		case 3:
		{
			func = SinusoidActivation;
			derivative = SinusoidDerivative;
		}
	}

	//initialise the network
	MLP mlp(Data.TrainingFeatures[0].size());
	for (int i = 0; i < layerLoops; ++i)
	{	
		int n = std::max(2,20-i);
		mlp.AddLayer(Layer(n,func,derivative));
	}
	//add the final layer -- always use logit since is a 1D classifier
	mlp.AddLayer(Layer(1,LogisticActivation,LogisticDerivative));

	
	//Begin training -- loop over different epoch lengths to show sequential improvements
	for (int epoch = 2; epoch < 1e5; epoch*=5)
	{
		mlp.Train(Data.TrainingFeatures,Data.TrainingTruths,epoch);
		DisplayProgress(mlp,xs,ys);
	}
}
void SimplePlot()
{
	LearningData Data = SynthesiseData(10000);
	std::vector<std::vector<double>> xs(2);
	std::vector<std::vector<double>> ys(2);
	for (int i =0; i < Data.TrainingFeatures.size(); ++i)
	{
		int v = Data.TrainingTruths[i][0];
		xs[v].push_back(Data.TrainingFeatures[i][0]);
		ys[v].push_back(Data.TrainingFeatures[i][1]);
	}
	JSL::gnuplot gp;
	gp.Scatter(std::vector<double>{0},std::vector<double>{0},JSL::LineProperties::ScatterType(JSL::Dot),JSL::LineProperties::PenSize(0));
	gp.Scatter(xs[0],ys[0],JSL::LineProperties::ScatterType(JSL::Dot),JSL::LineProperties::PenSize(3));
	gp.Scatter(xs[1],ys[1],JSL::LineProperties::ScatterType(JSL::Dot),JSL::LineProperties::PenSize(3));
	gp.SetPalette("gray");
	gp.SetXRange(-0.6,0.6);
	gp.SetYRange(-0.6,0.6);

	gp.Show();
}
int main(int argc, char**argv)
{
	JSL::Argument<int> PlottingShape(0,"mode",argc,argv); //changes the generation of data: 0 = donut, 1 = XOR, 2 = pokeball, 3= silly sine shape
	JSL::Argument<int> Layers(0,"layer",argc,argv); // changes the number of layers in the network (with 0 = a single output layer)
	JSL::Toggle Test(false,"test",argc,argv); //decativates the network, just plots the target data
	JSL::Argument<int> Activation(0,"activation",argc,argv); //changes the activation function, 0 = ReLu, 1 = Sigmoir, 2 = Linear, 3 = sin
	JSL::Argument<int> DataCount(10000,"N",argc,argv); // chanegs the amount of data used to evaluate the network
	activation = Activation;
	mode = PlottingShape;
	layerLoops = Layers;

	if (!Test)
	{	
		MLP_Demonstrate(DataCount);	
	}
	else
	{
		SimplePlot();
	}
}