#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <random>
#include "activationFunctions.h"
int rSeed = time(NULL);
std::mt19937 nodeRandomiser(rSeed);

//a node is a single weight vector - it performs dot products, activation funcs and updates itself
class Node
{
	public:
		int Dimension;
		std::vector<double> Weights;
		double Y;
		double dLdY;

		//various constructors
		Node()
		{
			internal_ActivationFunction = ReLUActivation;
			internal_ActivationDerivative = ReLUDerivative;
		}
		Node(int Dim) : Dimension(Dim)
		{
			Weights.resize(Dim+1);
			internal_ActivationFunction = ReLUActivation;
			internal_ActivationDerivative = ReLUDerivative;
		}
		Node(double (*activate)(double x),double (*derivative)(double x))
		{
			internal_ActivationFunction = activate;
			internal_ActivationDerivative = derivative;
		}
		Node(int Dim, double (*activate)(double x),double (*derivative)(double x)) : Dimension(Dim)
		{
			internal_ActivationFunction = activate;
			internal_ActivationDerivative = derivative;
			Weights.resize(Dim+1);
			
		}

		//initialisation function -- randomly sets weights between -5 and 5
		void Initialise(int dim)
		{
			Dimension = dim;
			Weights.resize(dim+1);
			
			for (int i = 0; i <= Dimension; ++i)
			{
				double r = 10*( (double)nodeRandomiser()/nodeRandomiser.max()-0.5);
				Weights[i] = r;
			}
		}

		//public access for activation functions
		double ActivationFunction()
		{
			return internal_ActivationFunction(Y);
		}
		double ActivationDerivative()
		{
			return internal_ActivationDerivative(Y);
		}

		//performs the dot product, stores it in Y, and then returns the activation function
		double Operate(const std::vector<double> & input)
		{
			Y = Weights[0];
			for (int i =0; i < Dimension; ++i)
			{
				Y += Weights[i+1] * input[i];
			}
			return ActivationFunction();
		}

		//applies the computed steps into this nodes weights
		int Update(const std::vector<double> & steps, int startIdx)
		{
			for (int i =0; i <= Dimension; ++i)
			{
				Weights[i] += steps[i+startIdx];
			}
			return Dimension+1;
		}
	private:
		double (*internal_ActivationFunction)(double x);
		double (*internal_ActivationDerivative)(double x);
};

//a layer holds a set of nodes. For simplicity, assume all nodes in a layer have the same activation function
class Layer
{
	public:
		std::vector<Node> Nodes;
		int NodeCount;
		int Dimension;
		std::vector<double> LayerVector; //value of X_n which is output
		
		//various constructors
		Layer(int nNodes)
		{
			NodeCount = nNodes;
			Nodes.resize(nNodes,Node());
			LayerVector.resize(nNodes);
		}
		Layer(int nNodes, double (*activate)(double x),double (*derivative)(double x))
		{
			NodeCount = nNodes;
			Nodes.resize(nNodes,Node(activate,derivative));
			LayerVector.resize(nNodes);
		}
		Layer(int nNodes, int nDim)
		{
			NodeCount = nNodes;
			Nodes.resize(nNodes,Node());
			LayerVector.resize(nNodes);
			Initialise(nDim);
		}

		//why initialise separately to the construction? Answer: node dimension is set by the previous layer, so can't know until inserted into network
		void Initialise(int nDim)
		{
			Dimension = nDim;
			for (int i = 0; i < NodeCount; ++i)
			{
				Nodes[i].Initialise(nDim);
			}
		}
		
		//this is the main forward-operation -- takes the downward layer vector, and passess it through the nodes
		void Process(const std::vector<double> & input)
		{
			for (int i = 0; i < NodeCount; ++i)
			{
				LayerVector[i] = Nodes[i].Operate(input);
			}
		}

		//passes update information ot the nodes
		int Update(const std::vector<double> & stepSize, int start)
		{
			for (int n = 0; n < NodeCount; ++n)
			{
				start += Nodes[n].Update(stepSize,start);
			}
			return start;
		};
		
};

class MLP
{
	public:
		const int Dimension;
		MLP(int dim) : Dimension(dim){};
		~MLP(){//custom destructor needed if you do a pure linked-list implementation. I simplified to vectors 
		}

		//adds a layer, connects into series and then initialises it
		void AddLayer(Layer layer)
		{
			int idx = Layers.size();
			int d = Dimension; //first layer in network uses input vetcor as dimension
			if (idx > 0)
			{
				d = Layers[idx-1].NodeCount;//else use previous layer size
			}
			layer.Initialise(d);
			Layers.push_back(layer);
			++LayerCount;
		}
		
		std::vector<double> Process(std::vector<double> in)
		{
			//pass the input vector sequentially through the network
			Layers[0].Process(in); 
			for (int i =1; i < LayerCount; ++i)
			{
				Layers[i].Process(Layers[i-1].LayerVector);
			}
			return Layers[LayerCount-1].LayerVector;
		}

		void Train(const std::vector<std::vector<double>> & features, const std::vector<std::vector<double>> & values)
		{
			Train(features,values,1000);
		}
		void Train(const std::vector<std::vector<double>> & features, const std::vector<std::vector<double>> & values, int epochs)
		{
			int B = features.size()/2;
			int featuresPerBatch = features.size()/B;
			std::vector<std::vector<double>> batchFeatures(featuresPerBatch);
			std::vector<std::vector<double>> batchValues(featuresPerBatch);

			std::vector<int> idx(features.size());
			std::iota(idx.begin(),idx.end(),0);
			int D = 0;
			nodeRandomiser = std::mt19937(0);
			for (int l = 0; l < LayerCount; ++l)
			{
				Layers[l].Initialise(Layers[l].Dimension);
				// for (int j = 0; j < Layers[l].NodeCount)
				D += Layers[l].NodeCount * (Layers[l].Dimension+1);
				// std::cout << "Layer " << l << " has " << Layers[l].NodeCount << " nodes of d = " << Layers[l].Dimension + 1 << std::endl;
			}

			std::vector<double> gradient(D,0.0);
			std::vector<double> stepSize(D,0.0);
			std::vector<double> memory(D,0.0);
			std::vector<double> second(D,0.0);
			double alpha = 0.001;
			double b1 = 0.9;
			double b2 = 0.999;
			double oldVal = -1e300;
			int oldInc = 0;
			std::random_device rd;
			std::mt19937 g(0);
			double decayFactor = 0.5;
			double harness = 1e-2;
			int recoverEpoch = std::max(5.0,epochs * 0.05);
			// std::cout << recoverEpoch << "   " << featuresPerBatch << std::endl;
			double harnessAccel = pow(1.0/harness,1.0/(recoverEpoch*B));
			int downShiftRun = std::max(10.0,-log(B)/log(decayFactor));
			// exit(2);
			double L = 0;
			double memoryL = 1;
			int rein = 0;
			double prevEpochL = 0;
			for (int n  =0; n < epochs; ++n)
			{
				std::shuffle(idx.begin(),idx.end(),g);
				L = 0;
				for (int b = 0; b < B; ++b)
				{
					for (int i = 0; i < featuresPerBatch; ++i)
					{
						int id = featuresPerBatch * b + i;
						batchFeatures[i] = features[idx[id]];
						batchValues[i] = values[idx[id]];
					}
				// FillGradients(gradient,features,values);
					double newVal = TrainingCompute(gradient,batchFeatures,batchValues);
					L += newVal;
					double g = 0;
					for (int i = 0; i < gradient.size(); ++i)
					{
						g += gradient[i] * gradient[i];
					
					}
					g = sqrt(g);
					double c1 = 1.0/(1.0 - pow(b1,n+1));
					double c2 = 1.0/(1.0 - pow(b2,n+1));
					for (int i = 0; i < gradient.size(); ++i)
					{
						memory[i] = b1 * memory[i] + (1.0 -b1) * gradient[i];
						second[i] = b2 * second[i] + (1.0 - b2) * gradient[i] * gradient[i];
						stepSize[i] = harness * alpha * memory[i]*c1 / (sqrt(second[i]*c2 + 1e-7));
						// if (i == gradient.size()-1)
						// {
						// 	std::cout << stepSize[i] << "   " << memory[i] << "  " << second[i] << "   " << harness << std::endl;
						// }
					}
					
					if (n==0 && b==0)
					{
						memoryL = newVal;
					}
					else
					{
						memoryL = b2 * memoryL + (1.0 - b2) * newVal;
						
					}
					if (newVal < 1 * memoryL)
					{
						harness = std::min(1.0,harnessAccel*harness);
					}		
					
					if (std::isnan(newVal))
					{
						std::cout << "ISNAN ERROR " << "  " << harness << std::endl;
						exit(2);
					}
				
					oldVal = newVal;
					// alpha *=0.85;
					// alpha = std::max(3e-3,alpha);
					int start = 0;
					for (int l = 0; l < LayerCount; ++l)
					{
						start = Layers[l].Update(stepSize,start);
					}

					// if (epochs)
				}
				

				if (n > 0)
				{
					if (abs(L) > abs(prevEpochL))
					{
						rein+=2;

						if (rein > 10)
						{
							double dec = 1e-1;
							B = std::max(B*0.9,1.0);
							harness = std::max(1e-1,dec*harness);
							harnessAccel = std::min(pow(1.0/dec,1.0/(30)),1.05);
							rein = 0;
							memoryL = L;
						}
					}
					else
					{
						rein = std::max(0,rein-1);
					}
				}
				prevEpochL = L;
			}
			std::cout << epochs << "-epoch training got an accuracy of " << TrainingError(features,values)/features.size() << " harness at " << harness << " (" << harnessAccel << ") and minibatches " << B << std::endl;
			
		}

		double TrainingError(const std::vector<std::vector<double>> & features, const std::vector<std::vector<double>> & values)
		{
			double L = 0;
			for (int i = 0; i < features.size(); ++i)
			{
				auto predict = Process(features[i]);
				// double d = 0;
				for (int j = 0; j < values[i].size(); ++j)
				{
					double d = predict[j] - values[i][j];
					L += d * d;
				}
			}
			return -0.5 * L;
		}
	
	private:
		int LayerCount = 0;
		std::vector<Layer> Layers;
		int LayerIdx(int layer, int node, int dimension)
		{
			int v =0;
			for (int l = 0; l < layer; ++l)
			{
				v += Layers[l].NodeCount * (Layers[l].Dimension + 1);
			}
			return v + node * (Layers[layer].Dimension + 1) + dimension;
		}

		//passes info forward through the network, then backpropagates to get derivatives
		double TrainingCompute(std::vector<double> & grads, const std::vector<std::vector<double>> & features, const std::vector<std::vector<double>> & values)
		{	
			double L = 0;
			std::fill(grads.begin(),grads.end(),0.0);

			//iterate over data
			for (int i = 0; i < features.size(); ++i)
			{
				//pass forward, then compute loss function
				auto predict = Process(features[i]);
				double sqErr = 0;
				for (int j = 0; j < values[i].size(); ++j)
				{
					double d = predict[j] - values[i][j];
					sqErr += d*d;
					L += d * d;
				}
			
				if (sqErr > 0.01) //hacky speedup/regularizer: only update weights if prediction wrong by more than 2%
				{
					//now onto derivatives: start at final layer, then work backwards
					for (int layer = LayerCount -1; layer >= 0; --layer)
					{
						Layer & currentLayer = Layers[layer];
						const std::vector<double> * previousPosition = &features[i];
						if (layer > 0)
						{
							previousPosition = &Layers[layer-1].LayerVector;
						}
						
						for (int node = 0; node < currentLayer.NodeCount; ++node)
						{
							Node & currentNode = currentLayer.Nodes[node];
							if (layer == LayerCount - 1)
							{
								//final layer derivative
								currentNode.dLdY = ((values[i][node] - predict[node]) ) * currentNode.ActivationDerivative();
							}
							else
							{
								Layer & upperLayer = Layers[layer+1];
								currentNode.dLdY = 0;

								//chain rule!
								for (int k = 0; k < upperLayer.NodeCount; ++k)
								{
									currentNode.dLdY += upperLayer.Nodes[k].Weights[node+1] * upperLayer.Nodes[k].dLdY;
								}
								currentNode.dLdY *= currentNode.ActivationDerivative();

							}
							int nidx = LayerIdx(layer,node,0);

							//some silly index hacking to get everything into the right place
							grads[nidx] += currentNode.dLdY;
							for (int b = 1; b <= currentNode.Dimension; ++b)
							{
								grads[nidx +b] += currentNode.dLdY * previousPosition[0][b-1];
							}
						}
					}
				}

			}	
			return -0.5 * L;
		}
};