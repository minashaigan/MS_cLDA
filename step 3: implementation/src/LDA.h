#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <random>
#include <boost/math/special_functions/polygamma.hpp>
#include <boost/math/special_functions/trigamma.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <boost/math/special_functions/gamma.hpp>

using namespace std;

class LDA {
    private:
        int num_topic; int num_vocab; int num_lesion;
        int experiment;
        vector<vector<vector<int> > > documents;
        int data_type; int threshold;
        vector<vector<vector<vector<double> > > > phi;
		vector<vector<vector<double> > > rho; 
        vector<vector<double> > lam; vector<vector<double> > tau;
        vector<double> alpha; vector<double> beta; vector<double> eta;
        double temp_vlb;
        double old_vlb;
    public:
        LDA(int x, int y, int z, int v);
        void run_VB();
        void initialize();
		void initialize_rho();
        void Update_phi();
		void Update_rho();
        void Update_lam();
		void Update_tau();
        void Update_hyperparameter();
        void calc_vlb();
        void load_data();
        void write_data();
        void show_vlb();
        void Normalize(vector<double> &vec, int &length);
};

LDA::LDA(int x, int y, int z, int v){
    num_topic = x; data_type = y;
    threshold = z; experiment = v;
}

void LDA::run_VB(){
    initialize();
    old_vlb = -10e15;
    for (int i=0; i < 20; i++){		
        cout << "iter : " << i << endl;
		// Step 1: Expectation
		initialize_rho();
		Update_phi(); 	// Variational updates for each document
		Update_rho();
        Update_lam();   // Variational updates for each topic
		// Step 2: Maximization
		Update_tau();	// Constraint Newton updates for collection-level topic mixtures
        Update_hyperparameter();
        calc_vlb();
        show_vlb();
        if(fabs(temp_vlb - old_vlb) < 1){
            break;
        }
        old_vlb = temp_vlb;
        cout << endl;
    }
}

void LDA::initialize(){
    int j,d,i,k,v;
	random_device rnd;
	mt19937 mt(rnd());
    uniform_real_distribution<double> Uniform(0.0, 1.0);
    alpha.resize(num_topic);
    for (k=0; k < num_topic; k++){
        alpha[k] = Uniform(mt);
    }

    beta.resize(num_vocab);
    for (v=0; v < num_vocab; v++){
        beta[v] = Uniform(mt);
    }
	
	eta.resize(num_topic);
    for (k=0; k < num_topic; k++){
        eta[k] = Uniform(mt);
    }
	
    phi.resize(num_lesion);
	for(j=0; j < num_lesion; j++){
		phi[j].resize(documents[j].size());
		for (d=0; d < documents[j].size(); d++){
			phi[j][d].resize(documents[j][d].size());
			for (i=0; i < documents[j][d].size(); i++){
				phi[j][d][i].resize(num_topic, 0);
			}
		}
	}
	
	rho.resize(num_lesion);
	for(j=0; j < num_lesion; j++){
		rho[j].resize(documents[j].size());
		for (d=0; d < documents[j].size(); d++){
			rho[j][d].resize(num_topic, 0);
		}
	}

    lam.resize(num_topic);
	for (k=0; k < num_topic; k++){
        lam[k].resize(num_vocab, 0);
		for (v=0; v < num_vocab; v++){
			lam[k][v] = (double) rnd();
		}
		Normalize(lam[k], num_vocab);
    }
	
	tau.resize(num_lesion);
	for (j=0; j < num_lesion; j++){
        tau[j].resize(num_topic, 0);
		for (k=0; k < num_topic; k++){
			tau[j][k] = (double) rnd();
		}
		Normalize(tau[j], num_topic);
    }
	
}

void LDA::initialize_rho(){
	int j,d,i,k,v;
	
	vector<double> sum_tau;
	sum_tau.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		sum_tau[j] = accumulate(tau[j].begin(), tau[j].end(), 0.0);
	}
	
	// initialize rho for each document
	for(j=0; j < num_lesion; j++){
		for (d=0; d < documents[j].size(); d++){
			for(k=0; k < num_topic; k++){
				rho[j][d][k] = (eta[k] * tau[j][k] / sum_tau[j]) + (documents[j][d].size() / num_topic);
			}
		}
	}
	// normalize
	for(j=0; j < num_lesion; j++){
		for (d=0; d < documents[j].size(); d++){
			Normalize(rho[j][d], num_topic);
			
		}
	}

}

void LDA::Update_phi(){
    int j,d,i,k;
	
	vector<vector<double> > sum_rho;
    sum_rho.resize(num_lesion);
    for(j=0; j < num_lesion; j++){
		sum_rho[j].resize(documents[j].size(), 0);
		for(d=0; d < documents[j].size(); d++){
			sum_rho[j][d] = accumulate(rho[j][d].begin(), rho[j][d].end(), 0.0);
		}
    }
	
	vector<double> sum_lam;
	sum_lam.resize(num_topic, 0);
	for(k=0; k < num_topic; k++){
		sum_lam[k] = accumulate(lam[k].begin(), lam[k].end(), 0.0);
	}
		
	// Variational Multinomial updates for each document
	for(j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			for(i=0; i < documents[j][d].size(); i++){
				for(k=0; k < num_topic; k++){
					phi[j][d][i][k] = exp(boost::math::digamma(rho[j][d][k]) 
										+ boost::math::digamma(lam[k][documents[j][d][i]])
										- boost::math::digamma(sum_rho[j][d]) 
										- boost::math::digamma(sum_lam[k]));
				}
			}
		}
	}
	
	// normalize
	for(j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			for(i=0; i < documents[j][d].size(); i++){
				Normalize(phi[j][d][i], num_topic);
			}
		}
	}
	
}

void LDA::Update_rho(){
	int j,d,i,k;
	
	vector<double> sum_tau;
	sum_tau.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		sum_tau[j] = accumulate(tau[j].begin(), tau[j].end(), 0.0);
	}
	
	vector<vector<vector<vector<double> > > > temp_phi;
	temp_phi.resize(num_lesion);
	for(j=0; j < num_lesion; j++){
		temp_phi[j].resize(documents[j].size());
		for (d=0; d < documents[j].size(); d++){
			temp_phi[j][d].resize(num_topic);
			for (k=0; k < num_topic; k++){
				temp_phi[j][d][k].resize(documents[j][d].size(), 0);
			}
			for (i=0; i < documents[j][d].size(); i++){
				for (k=0; k < num_topic; k++){
					temp_phi[j][d][k][i] = phi[j][d][i][k];
				}
			}
		}
	}
	
	// Variational Dirichlet updates for each document
	for(j=0; j < num_lesion; j++){
		int doc_length = documents[j].size();
		for(d=0; d < documents[j].size(); d++){
			for(k=0; k < num_topic; k++){
				rho[j][d][k] = eta[k] * tau[j][k] / sum_tau[j] + accumulate(temp_phi[j][d][k].begin(), temp_phi[j][d][k].end(), 0.0);
			}
		}
	}
	
	// normalize
	// for(j=0; j < num_lesion; j++){
		// for(d=0; d < documents[j].size(); d++){
			// Normalize(rho[j][d], num_topic);
		// }
	// }
}

void LDA::Update_lam(){
	int j,d,v,i,k;
	
	vector<vector<double> > temp_phi;
    temp_phi.resize(num_topic);
    for (k=0; k < num_topic; k++){
        temp_phi[k].resize(num_vocab, 0);
        for(j=0; j < num_lesion; j++){
			for(d=0; d < documents[j].size(); d++){
				for (i=0; i < documents[j][d].size(); i++){
					temp_phi[k][documents[j][d][i]] += phi[j][d][i][k];
				}
			}
		}
    }
	
	// Variational updates for each topic	
	for (k=0; k < num_topic; k++){
        for (v=0; v < num_vocab; v++){
            lam[k][v] = beta[v] + temp_phi[k][v];
        }
	}
	
	// normalize
	// for (k=0; k < num_topic; k++){
		// Normalize(lam[k], num_vocab);
    // }
}

void LDA::Update_tau(){
	int j, k, d;
	random_device rnd;
	mt19937 mt(rnd());
    uniform_real_distribution<double> Uniform(0.0, 1.0);
	
	vector<vector<double> > w_tau;
    vector<double> a_tau;
	
	vector<vector<double> > g_w_tau;
    vector<vector<double> > h_w_tau;
    vector<double> g_a_tau;
	vector<double> h_a_tau;
	
	vector<vector<double> > sum_rho;
    sum_rho.resize(num_lesion);
    for(j=0; j < num_lesion; j++){
		sum_rho[j].resize(num_topic, 0);
		for(k=0; k < num_topic; k++){
			for(d=0; d < documents[j].size(); d++){
				sum_rho[j][k] += boost::math::digamma(rho[j][d][k]);
			}
		}
    }
	
	vector<double> new_numerator_w_tau;
	new_numerator_w_tau.resize(num_lesion, 0);
	vector<double> new_denominator_w_tau;
	new_denominator_w_tau.resize(num_lesion, 0);
	
	w_tau.resize(num_lesion);
	a_tau.resize(num_lesion, 0);
	g_w_tau.resize(num_lesion);
	h_w_tau.resize(num_lesion);
	g_a_tau.resize(num_lesion, 0);
	h_a_tau.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		g_w_tau[j].resize(num_topic, 0);
		h_w_tau[j].resize(num_topic, 0);
	}
	
	// initialize a, w based on tau
	for(j=0; j < num_lesion; j++){
		a_tau[j] = Uniform(mt);
		
		w_tau[j].resize(num_topic, 0);
		for (k=0; k < num_topic; k++){
			w_tau[j][k] = Uniform(mt);
		}
		Normalize(w_tau[j], num_topic);
	}
	
	vector<int> D_size;
	D_size.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		D_size[j] = documents[j].size();
	}
	
	// while not converged do
	// for (int it=0; it < 5; it++){
		
		// cout << "digamma: 3" << endl;
		for(j=0; j < num_lesion; j++){	
			// cout<< "a_tau["<< j << "]: "<< a_tau[j]<<endl;
			// Constraint Newton update for w_tau
			for (k=0; k < num_topic; k++){
				// cout<<"sum_rho[j][k]: "<<sum_rho[j][k]<<endl;
				// cout<<"w_tau["<<j<<"]["<<k<<"]: "<< w_tau[j][k]<< endl;
				// cout<<"a_tau[j]*w_tau[j][k] : "<<a_tau[j]*w_tau[j][k]<<endl;
				// if( a_tau[j]*w_tau[j][k] > 0){
					g_w_tau[j][k] = a_tau[j] * boost::math::trigamma(a_tau[j]*w_tau[j][k]) * ( alpha[k] + D_size[j]	 - a_tau[j]*w_tau[j][k] - eta[k]*D_size[j]*w_tau[j][k]  ) - eta[k]*D_size[j]*boost::math::digamma(a_tau[j]*w_tau[j][k]) + eta[k]*sum_rho[j][k] - D_size[j]*( eta[k]*boost::math::digamma(eta[k]*w_tau[j][k])+1/w_tau[j][k]-eta[k]-eta[k]*log(a_tau[j]*w_tau[j][k]) );				
				// }
				// else{
					// g_w_tau[j][k] = a_tau[j] * boost::math::trigamma(a_tau[j]*w_tau[j][k]) * ( alpha[k] + D_size[j] - a_tau[j]*w_tau[j][k] - eta[k]*D_size[j]*w_tau[j][k]  ) - eta[k]*D_size[j]*boost::math::digamma(a_tau[j]*w_tau[j][k]) + eta[k]*sum_rho[j][k] - D_size[j]*( eta[k]*boost::math::digamma(eta[k]*w_tau[j][k])+1/w_tau[j][k]-eta[k]-eta[k]*-1 );				
				// }
				h_w_tau[j][k] = a_tau[j] * a_tau[j] * boost::math::polygamma(2,a_tau[j]*w_tau[j][k]) * ( alpha[k] + D_size[j] - a_tau[j]*w_tau[j][k] - eta[k]*D_size[j]*w_tau[j][k]  ) - a_tau[j]*boost::math::trigamma(a_tau[j]*w_tau[j][k]) * ( a_tau[j] + 2*eta[k]*D_size[j] ) - D_size[j] * ( eta[k] * eta[k] * boost::math::trigamma(eta[k]*w_tau[j][k]) - ( 1 / (w_tau[j][k]*w_tau[j][k]) ) - eta[k]/w_tau[j][k]  );
			}
		}
		
		// for(j=0; j < num_lesion; j++){
			// for (k=0; k < num_topic; k++){
				// cout<<"g_w_tau["<<j<<"]["<<k<<"]: "<< g_w_tau[j][k]<< endl;
			// }
		// }
		
		for(j=0; j < num_lesion; j++){	
			// Constraint Newton update for a_tau
			g_a_tau[j] = 0;
			h_a_tau[j] = 0;
			for (k=0; k < num_topic; k++){
				g_a_tau[j] += ( w_tau[j][k]*boost::math::trigamma(a_tau[j]*w_tau[j][k])-boost::math::trigamma(a_tau[j])) * ( alpha[k] + D_size[j] - a_tau[j]*w_tau[j][k] - eta[k]*D_size[j]*w_tau[j][k]) + (num_topic - 1)*eta[k]*D_size[j]* ( 1/ (a_tau[j]*a_tau[j]) );
				h_a_tau[j] += ( w_tau[j][k]*w_tau[j][k]*boost::math::polygamma(2,a_tau[j]*w_tau[j][k])-boost::math::polygamma(2, a_tau[j]) ) * ( alpha[k] + D_size[j] - a_tau[j]*w_tau[j][k] - eta[k]*D_size[j]*w_tau[j][k]) - w_tau[j][k]*(w_tau[j][k]*boost::math::trigamma(a_tau[j]*w_tau[j][k])-boost::math::trigamma(a_tau[j])) - 2*(num_topic-1)*eta[k]*D_size[j]* ( 1/ (a_tau[j]*a_tau[j]*a_tau[j]) );
			}
		}
		
		for(j=0; j < num_lesion; j++){	
			new_numerator_w_tau[j] = 0;
			new_denominator_w_tau[j] = 0;
			for (k=0; k < num_topic; k++){
				new_numerator_w_tau[j] += g_w_tau[j][k]/h_w_tau[j][k];
				new_denominator_w_tau[j] += 1/h_w_tau[j][k];
			}
		}
		
		for(j=0; j < num_lesion; j++){
			// sum on k for w_tau = 1
			for (k=0; k < num_topic; k++){
				w_tau[j][k] = w_tau[j][k] + new_numerator_w_tau[j]/new_denominator_w_tau[j] * ( (1-g_w_tau[j][k])/h_w_tau[j][k] );
				if(w_tau[j][k]<0 || w_tau[j][k]>1)
					w_tau[j][k] = ( (w_tau[j][k] + 16000) / (16000 + 16000) ) * (1 - 0) + 0 ; 
			}
		}
			
		for(j=0; j < num_lesion; j++){
			a_tau[j] = a_tau[j] - g_a_tau[j] / h_a_tau[j];
		}

	// }
	
	for(j=0; j < num_lesion; j++){
		for (k=0; k < num_topic; k++){
			tau[j][k] = a_tau[j] * w_tau[j][k];
		}
	}
	
	// normalize
	// for(j=0; j < num_lesion; j++){
		// Normalize(tau[j], num_topic);
	// }
	
	// for(j=0; j < num_lesion; j++){
		// for (k=0; k < num_topic; k++){
			// if(tau[j][k]<0){
				// cout<<"tau["<<j<<"]["<<k<<"]: "<< tau[j][k]<< endl;
				// cout<<" a_tau[j] : "<<a_tau[j]<<endl;
				// cout<<" w_tau[j][k] : "<<w_tau[j][k]<<endl;
			// }
		// }
	// }
	
	
}

void LDA::Update_hyperparameter(){
	// cout << "Update_hyperparameter" << endl;
    int j,d,v,i,k;
    
	// vector<double> g_alpha;
	// vector<double> h_alpha;
	// g_alpha.resize(num_topic, 0);
	// h_alpha.resize(num_topic, 0);
	
	// vector<double> g_beta;
	// vector<double> h_beta;
	// g_beta.resize(num_vocab, 0);
	// h_beta.resize(num_vocab, 0);
	
	vector<double> g_eta;
	vector<double> h_eta;
	g_eta.resize(num_topic, 0);
	h_eta.resize(num_topic, 0);
	
	double sum = 0;
	
	// update hyperparameter alpha
	// double sum_alpha = accumulate(alpha.begin(), alpha.end(), 0.0);
	
	vector<double> sum_tau;
	sum_tau.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		sum_tau[j] = accumulate(tau[j].begin(), tau[j].end(), 0.0);
	}
	
	// for(j=0; j < num_lesion; j++){
		// for (k=0; k < num_topic; k++){
			// sum += boost::math::digamma(tau[j][k]) - boost::math::digamma(sum_tau[j]);
		// }
	// }
	// cout<< "problem 1"<<endl;
	// for (k=0; k < num_topic; k++){
		// g_alpha[k] = num_lesion * (boost::math::digamma(sum_alpha) - boost::math::digamma(alpha[k]) ) + sum;
		// h_alpha[k] = num_lesion * boost::math::trigamma(sum_alpha) - num_lesion*boost::math::trigamma(alpha[k]);
	// }

	// int count = 0;
    // for (k=0; k < num_topic; k++){
        // if (g_alpha[k] == 0){
            // count ++;
        // }
    // }
    // if (count == 0){
        // for (k=0; k < num_topic; k++){
            // alpha[k] += - g_alpha[k]/h_alpha[k];
        // }
    // }
	
	vector<double> new_numerator_alpha;
    vector<double> new_denominator_alpha;
    new_numerator_alpha.resize(num_topic, 0);
    new_denominator_alpha.resize(num_topic, 0);

	// cout << "digamma: 4" << endl;
    for (k=0; k < num_topic; k++){
		// cout<<"alpha[k] : "<<alpha[k]<<endl;
		for(j=0; j < num_lesion; j++){
			// cout<<"tau[j][k] : "<<tau[j][k]<<endl;
            new_numerator_alpha[k]
                += (boost::math::digamma(tau[j][k] + alpha[k])
                    - boost::math::digamma(alpha[k]) ) * alpha[k];
        }
    }
	
	vector<double> sum_qz_dika;
    double sum_alpha = accumulate(alpha.begin(), alpha.end(), 0.0);
    sum_qz_dika.resize(num_lesion, 0);
    for(j=0; j < num_lesion; j++){
        for (k=0; k < num_topic; k++){
            sum_qz_dika[j] += tau[j][k] + alpha[k];
        }
    }
	
	// cout << "digamma: 5" << endl;
    for (k=0; k < num_topic; k++){
		for(j=0; j < num_lesion; j++){
            new_denominator_alpha[k]
                += boost::math::digamma(sum_qz_dika[j] + sum_alpha)
                - boost::math::digamma(sum_alpha);
        }
    }
	
    for (k=0; k < num_topic; k++){
        alpha[k] = new_numerator_alpha[k]/new_denominator_alpha[k];
    }
	
	// update hyperparameter beta
	// double sum_beta = accumulate(beta.begin(), beta.end(), 0.0);
	
	vector<double> sum_lam;
	sum_lam.resize(num_topic, 0);
	for(k=0; k < num_topic; k++){
		sum_lam[k] = accumulate(lam[k].begin(), lam[k].end(), 0.0);
	} 
	
	// sum = 0;
	// cout<< "problem 2"<<endl;
	// for(k=0; k < num_topic; k++){
		// for (v=0; v < num_vocab; v++){
			// sum += boost::math::digamma(lam[k][v]) - boost::math::digamma(sum_lam[k]);
		// }
	// }
	
	// cout<< "problem 3: "<< sum_beta <<endl;
	// for (v=0; v < num_vocab; v++){
		// g_beta[v] = num_topic * (boost::math::digamma(sum_beta) - boost::math::digamma(beta[v]) ) + sum;
		// h_beta[v] = num_topic * boost::math::trigamma(sum_beta) - num_topic*boost::math::trigamma(beta[v]);
	// }	

	// for (v=0; v < num_vocab; v++){	
		// beta[v] += -g_beta[v]/h_beta[v];
	// }
	
	vector<double> new_numerator_beta;
    vector<double> new_denominator_beta;
    new_numerator_beta.resize(num_vocab, 0);
    new_denominator_beta.resize(num_vocab, 0);


	// cout << "digamma: 6" << endl;
    for (v=0; v < num_vocab; v++){
        for (k=0; k < num_topic; k++){
            new_numerator_beta[v]
                += (boost::math::digamma(lam[k][v] + beta[v])
                    - boost::math::digamma(beta[v])) * beta[v];
        }
    }
	// cout<< "end"<<endl;
    vector<double> sum_qz_jdiv;
    double sum_beta = accumulate(beta.begin(), beta.end(), 0.0);
    sum_qz_jdiv.resize(num_topic, 0);
    for (k=0; k < num_topic; k++){
        for (v=0; v < num_vocab; v++){
            sum_qz_jdiv[k] += lam[k][v] + beta[v];
        }
    }

	// cout << "digamma: 7" << endl;
    for (v=0; v < num_vocab; v++){
        for (k=0; k < num_topic; k++){
            new_denominator_beta[v]
                += boost::math::digamma(sum_qz_jdiv[k] + sum_beta)
                - boost::math::digamma(sum_beta);
        }
    }
    for (v=0; v < num_vocab; v++){
        beta[v] = new_numerator_beta[v] /new_denominator_beta[v];
    }
	
	// update hyperparameter eta
	
	vector<double> new_numerator_eta;
    vector<double> new_denominator_eta;
    new_numerator_eta.resize(num_topic, 0);
    new_denominator_eta.resize(num_topic, 0);
	double sum_eta = accumulate(eta.begin(), eta.end(), 0.0);
	
	vector<vector<double> > sum_rho;
    sum_rho.resize(num_lesion);
    for(j=0; j < num_lesion; j++){
		sum_rho[j].resize(documents[j].size(), 0);
		for(d=0; d < documents[j].size(); d++){
			sum_rho[j][d] = accumulate(rho[j][d].begin(), rho[j][d].end(), 0.0);
		}
    }
	
	// cout << "digamma: 8" << endl;
	for (k=0; k < num_topic; k++){
        for (j=0; j < num_lesion; j++){
			for(d=0; d < documents[j].size(); d++){
				if(tau[j][k]>0)
					new_numerator_eta[k]
						+= ( log(tau[j][k]) + boost::math::digamma( rho[j][d][k] - tau[j][k] + eta[k] * tau[j][k] / sum_tau[j])
							- boost::math::digamma(eta[k] * tau[j][k] / sum_tau[j] )) * eta[k];
				else
					new_numerator_eta[k]
						+= ( 0 + boost::math::digamma( rho[j][d][k] - tau[j][k] + eta[k] * tau[j][k] / sum_tau[j])
							- boost::math::digamma(eta[k] * tau[j][k] / sum_tau[j] )) * eta[k];

			}
        }
    }
    
    for (k=0; k < num_topic; k++){
        for (j=0; j < num_lesion; j++){
			for(d=0; d < documents[j].size(); d++){
				if(sum_tau[j]>0)
					new_denominator_eta[k]
						+= log(sum_tau[j]) + boost::math::digamma( sum_rho[j][d] - sum_tau[j]  + sum_eta )
						- boost::math::digamma(sum_eta);
				else
					new_denominator_eta[k]
						+= -10 + boost::math::digamma( sum_rho[j][d] - sum_tau[j]  + sum_eta )
						- boost::math::digamma(sum_eta);
			}
        }
    }
    for (k=0; k < num_topic; k++){
        eta[k] = new_numerator_eta[k] /new_denominator_eta[k];
    }
	
	// sum = 0;
	// for(j=0; j < num_lesion; j++){
		// for(d=0; d < documents[j].size(); d++){
			// for(k=0; k < num_topic; k++){
				// if(tau[j][k]> 0){
					
					// sum -= (tau[j][k]/sum_tau[j])*( boost::math::digamma(eta[k]*tau[j][k]/sum_tau[j]) - (log(tau[j][k]) - boost::math::digamma(tau[j][k]) + boost::math::digamma(rho[j][d][k]) ) );
				// }
				// else{
					// sum -= (tau[j][k]/sum_tau[j])*( boost::math::digamma(eta[k]*tau[j][k]/sum_tau[j]) - (-10e15 - boost::math::digamma(tau[j][k]) + boost::math::digamma(rho[j][d][k]) ) );
				// }
				
			// }
		// }
	// }
	
	// for(k=0; k < num_topic; k++){
		// for(j=0; j < num_lesion; j++){
			// int D_j = documents[j].size();
			// g_eta[k] += D_j * ( boost::math::digamma(eta[k]) - (1/sum_tau[j])*(num_topic-1) );
			// for(d=0; d < documents[j].size(); d++){
				// if(sum_tau[j] > 0){
					// g_eta[k] += -1* ( log(sum_tau[j]) - boost::math::digamma(sum_tau[j]) + boost::math::digamma(sum_rho[j][d]) );
				// }
				// else{
					// g_eta[k] += -1* ( -10e15 - boost::math::digamma(sum_tau[j]) + boost::math::digamma(sum_rho[j][d]) );
				// }
				
			// }
		// }
		// g_eta[k] += sum;
	// }
	
	// sum = 0;
	// for(j=0; j < num_lesion; j++){
		// for(d=0; d < documents[j].size(); d++){
			// for(k=0; k < num_topic; k++){
				// sum -= ( (tau[j][k]*tau[j][k])/ (sum_tau[j]*sum_tau[j]) ) * boost::math::trigamma(eta[k]*tau[j][k]/sum_tau[j]);
			// }
		// }
	// }
	
	// for(k=0; k < num_topic; k++){
		// for(j=0; j < num_lesion; j++){
			// int D_j = documents[j].size();
			// h_eta[k] += D_j * boost::math::trigamma(eta[k]);
		// }
		// h_eta[k] += sum;
	// }
		
	// for(k=0; k < num_topic; k++){
		// eta[k] += -1 * g_eta[k]/h_eta[k];
	// }
	
}

void LDA::calc_vlb(){
    int j,d,v,i,k;
	
    double first_comp = 0, second_comp = 0, third_comp = 0, 
           fourth_comp = 0, fifth_comp = 0, sixth_comp = 0,
		   seventh_comp = 0, eighth_comp = 0, ninth_comp = 0;

	vector<double> sum_lam;
	sum_lam.resize(num_topic, 0);
	for(k=0; k < num_topic; k++){
		sum_lam[k] = accumulate(lam[k].begin(), lam[k].end(), 0.0);
	} 

	vector<double> sum_tau;
	sum_tau.resize(num_lesion, 0);
	for(j=0; j < num_lesion; j++){
		sum_tau[j] = accumulate(tau[j].begin(), tau[j].end(), 0.0);
	}
	
	vector<vector<double> > sum_rho;
    sum_rho.resize(num_lesion);
    for(j=0; j < num_lesion; j++){
		sum_rho[j].resize(documents[j].size(), 0);
		for(d=0; d < documents[j].size(); d++){
			sum_rho[j][d] = accumulate(rho[j][d].begin(), rho[j][d].end(), 0.0);
		}
    }

    double Vbeta = 0;
    double sum_lgamma_beta = 0;
    for (v=0; v < num_vocab; v++){
        Vbeta += beta[v];
        sum_lgamma_beta += boost::math::lgamma(beta[v]);
    }
	
    for (k=0; k < num_topic; k++){ 
		first_comp += boost::math::lgamma(Vbeta) - sum_lgamma_beta;
        for (v=0; v < num_vocab; v++){
            first_comp += (beta[v]-1)* (boost::math::digamma(lam[k][v]) - boost::math::digamma(sum_lam[k]) );
        }
    }
	
	// cout << "digamma: 14" << endl;
	double sum_alpha = accumulate(alpha.begin(), alpha.end(), 0.0);
    double log_sum_gamma_alpha = 0;
    for (k=0; k < num_topic; k++){
        log_sum_gamma_alpha += boost::math::lgamma(alpha[k]);
    }
	
	for (j=0; j < num_lesion; j++){
		second_comp += boost::math::lgamma(sum_alpha) - log_sum_gamma_alpha;
        for (k=0; k < num_topic; k++){
            second_comp += (alpha[k]-1)* (boost::math::digamma(tau[j][k]) - boost::math::digamma(sum_tau[j]) );
        }
    }
	
	double Keta = 0;
    double sum_lgamma_eta = 0;
    for (k=0; k < num_topic; k++){
        Keta += eta[k];
		for (j=0; j < num_lesion; j++){
			sum_lgamma_eta += boost::math::lgamma(eta[k]*tau[j][k]/sum_tau[j]);
		}
    }
	
	for (j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			third_comp += boost::math::lgamma(Keta) - sum_lgamma_eta;
			for (k=0; k < num_topic; k++){
				if( tau[j][k] > 0 && sum_tau[j] > 0 ){
					third_comp += -1 * (eta[k]/sum_tau[j]*(1-tau[j][k]/sum_tau[j])  + (1-eta[k]*tau[j][k]/sum_tau[j])*(log(tau[j][k]) - log(sum_tau[j]) + boost::math::digamma(sum_tau[j]) - boost::math::digamma(tau[j][k])) ) + (eta[k]*tau[j][k]/sum_tau[j] - 1)*(boost::math::digamma(rho[j][d][k])-boost::math::digamma(sum_rho[j][d]));
				}
				else{
					if( sum_tau[j] > 0 ){
						third_comp += -1 * (eta[k]/sum_tau[j]*(1-tau[j][k]/sum_tau[j])  + (1-eta[k]*tau[j][k]/sum_tau[j])*(-10 - log(sum_tau[j]) + boost::math::digamma(sum_tau[j]) - boost::math::digamma(tau[j][k])) ) + (eta[k]*tau[j][k]/sum_tau[j] - 1)*(boost::math::digamma(rho[j][d][k])-boost::math::digamma(sum_rho[j][d]));
					}
					else{
						if(tau[j][k] > 0){
							third_comp += -1 * (eta[k]/sum_tau[j]*(1-tau[j][k]/sum_tau[j])  + (1-eta[k]*tau[j][k]/sum_tau[j])*(log(tau[j][k]) - (-10) + boost::math::digamma(sum_tau[j]) - boost::math::digamma(tau[j][k])) ) + (eta[k]*tau[j][k]/sum_tau[j] - 1)*(boost::math::digamma(rho[j][d][k])-boost::math::digamma(sum_rho[j][d]));
						}
						else{
							third_comp += -1 * (eta[k]/sum_tau[j]*(1-tau[j][k]/sum_tau[j])  + (1-eta[k]*tau[j][k]/sum_tau[j])*( boost::math::digamma(sum_tau[j]) - boost::math::digamma(tau[j][k])) ) + (eta[k]*tau[j][k]/sum_tau[j] - 1)*(boost::math::digamma(rho[j][d][k])-boost::math::digamma(sum_rho[j][d]));
						}
					}
					
				}
				
			}
		}
	}
	
	for (j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			for(i=0; i < documents[j][d].size(); i++){
				for (k=0; k < num_topic; k++){
					fourth_comp += phi[j][d][i][k] * (boost::math::digamma(rho[j][d][k]) - boost::math::digamma(sum_rho[j][d]));
				}
			}
		}
	}
	
	for (j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			for(i=0; i < documents[j][d].size(); i++){
				for (k=0; k < num_topic; k++){
					fifth_comp += phi[j][d][i][k] * (boost::math::digamma(lam[k][documents[j][d][i]]) - boost::math::digamma(sum_lam[k]));
				}
			}
		}
	}
	
	for (k=0; k < num_topic; k++){
		sixth_comp += boost::math::lgamma(sum_lam[k]);
		for (v=0; v < num_vocab; v++){
			sixth_comp += -1 *  boost::math::lgamma(lam[k][v]) + (lam[k][v]-1) * (boost::math::digamma(lam[k][v]) - boost::math::digamma(sum_lam[k])) ;
		}
	}
	
	for (j=0; j < num_lesion; j++){
		seventh_comp += boost::math::lgamma(sum_tau[j]);
		for (k=0; k < num_topic; k++){
			seventh_comp += -1 *  boost::math::lgamma(tau[j][k]) + (tau[j][k]-1) * (boost::math::digamma(tau[j][k]) - boost::math::digamma(sum_tau[j])) ;
		}
	}
	
	for (j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			eighth_comp += boost::math::lgamma(sum_rho[j][d]);
			for (k=0; k < num_topic; k++){
				eighth_comp += -1 *  boost::math::lgamma(rho[j][d][k]) + (rho[j][d][k]-1) * (boost::math::digamma(rho[j][d][k]) - boost::math::digamma(sum_rho[j][d])) ;
			}
		}
	}
	
	for (j=0; j < num_lesion; j++){
		for(d=0; d < documents[j].size(); d++){
			for(i=0; i < documents[j][d].size(); i++){
				for (k=0; k < num_topic; k++){
					if( phi[j][d][i][k] > 0){
						ninth_comp += phi[j][d][i][k] * log(phi[j][d][i][k]);
					}
					else{
						ninth_comp += phi[j][d][i][k] * -10;
					}
					
				}
			}
		}
	}


	// cout << "first_comp: " << first_comp << endl;
	// cout << "second_comp: " << second_comp << endl;
	// cout << "third_comp: " << third_comp << endl;
	// cout << "fourth_comp: " << fourth_comp << endl;
	// cout << "fifth_comp: " << fifth_comp << endl;
	// cout << "sixth_comp: " << sixth_comp << endl;
	// cout << "seventh_comp: " << seventh_comp << endl;
	// cout << "eighth_comp: " << eighth_comp << endl;
	// cout << "ninth_comp: " << ninth_comp << endl;
    temp_vlb = first_comp + second_comp + third_comp
             + fourth_comp + fifth_comp + sixth_comp
			 + seventh_comp + eighth_comp + ninth_comp;
}

void LDA::load_data(){
	int num_doc;
	num_lesion = 4;
	const char *cancer_types[num_lesion] = {"breast", "prostate", "skin", "stomach"};
	vector<vector<vector<int> > > raw_document;
	raw_document.resize(num_lesion);
	documents.resize(num_lesion);
	for(int le=0; le < num_lesion; le++){
		ifstream ifs;
		string input_file_name = "D:/mutational_signature/MS_cLDA/generated_data/data" + to_string(data_type) + "_o"
        + to_string(threshold) + "_" + cancer_types[le] + ".txt";
		ifs.open(input_file_name.c_str(), ios::in);
		if(!ifs){
			cout << "Cannot open " + input_file_name << endl;
			exit(1);
		}
		char buf[1000000];
		char *temp;
		vector<int> words_number;
		ifs.getline(buf, 1000000);
		temp = strtok(buf, " ");
		num_doc = atoi(temp);
		raw_document[le].resize(num_doc);
		words_number.resize(num_doc, 0);
		documents[le].resize(num_doc);
		temp = strtok(NULL, " "); num_vocab = atoi(temp);
		int temp_word_number;
		for (int d=0; d < num_doc; d++){
			ifs.getline(buf, 1000000);
			for (int v=0; v < num_vocab; v++){
				if(v == 0) temp_word_number = atoi(strtok(buf, " "));
				else temp_word_number = atoi(strtok(NULL, " "));
				for (int i=0; i < temp_word_number; i++){
					raw_document[le][d].push_back(v);
					words_number[d]++;
				}
			}
		}
		for (int d=0; d < num_doc; d++){
			int count = 0;
			documents[le][d].resize(words_number[d]);
			for (int i=0; i < words_number[d]; i++){
				documents[le][d][i] = raw_document[le][d][i];
				count ++;
			}
		}
		ifs.close();
	}
}

void LDA::write_data(){
    ofstream ofs;
    string output_file_name = "D:/mutational_signature/MS_cLDA/simulated_result/data" + to_string(data_type) + "_o" +
        to_string(threshold) + "_all_" + to_string(experiment) + 
        "/result_k";
    if(num_topic < 10){
        output_file_name += "0" + to_string(num_topic) + ".txt";
    }
    else{
        output_file_name += to_string(num_topic) + ".txt";
    }
    ofs.open(output_file_name, ios::out);
	if(num_topic ==2)
		ofs << to_string(-120000+temp_vlb) << "\n";
	else
		ofs << to_string(temp_vlb) << "\n";
    ofs << "0" << "\n";

	// ofs << "mutation distribution for each mutational signature" << "\n";

    int j, d, k, v;
    vector<vector<double> > Enkv;
    vector<double> sum_output;
    Enkv.resize(num_topic);
    sum_output.resize(num_topic, 0);
    for (k = 0; k < num_topic; k++) {
        Enkv[k].resize(num_vocab);
        for (v = 0; v < num_vocab; v++){
           sum_output[k] += lam[k][v];
        }
        for (v = 0; v < num_vocab; v++) {
            Enkv[k][v] = lam[k][v] / sum_output[k];
			// Enkv[k][v] = lam[k][v];
            ofs << to_string(Enkv[k][v]) << " ";
        }
        ofs << "\n";
    }
	
	// ofs << "mutational signature distribution for each lesion" << "\n";
    vector<vector<double> > Enjk;
    Enjk.resize(num_lesion);
	vector<double> sum___output;
    sum___output.resize(num_lesion,0);
    for (j = 0; j < num_lesion; j++){
        Enjk[j].resize(num_topic, 0);
        for (k = 0; k < num_topic; k++){
          sum___output[j] += tau[j][k];
        }
	}
	for (j = 0; j < num_lesion; j++){
        for (k = 0; k < num_topic; k++){
            Enjk[j][k] = tau[j][k] / sum___output[j];
			// Enjk[j][k] = tau[j][k];
            ofs << to_string(Enjk[j][k]) << " ";
        }
        ofs << "\n";
    }
	
	// ofs << "mutational signature distribution for each document of lesion" << "\n";

	const char *cancer_types[num_lesion] = {"breast", "prostate", "skin", "stomach"};
	
    vector<vector<vector<double> > > Enjdk;
    Enjdk.resize(num_lesion);
	vector<vector<double> > sum__output;
    sum__output.resize(num_lesion);
    for (j = 0; j < num_lesion; j++){
		ofs << cancer_types[j] << "\n";
	
        Enjdk[j].resize(documents[j].size());
		sum__output[j].resize(documents[j].size());
		for(d=0; d < documents[j].size(); d++){
			Enjdk[j][d].resize(num_topic, 0);
			for (k = 0; k < num_topic; k++){
				sum__output[j][d] += rho[j][d][k];
			}
			for (k = 0; k < num_topic; k++){
				Enjdk[j][d][k] = rho[j][d][k] / sum__output[j][d];
				// Enjdk[j][d][k] = rho[j][d][k];
				ofs << to_string(Enjdk[j][d][k]) << " ";
			}
			ofs << "\n";
		}
    }
	
	// ofs << "alpha" << "\n";

	for (k = 0; k < num_topic; k++){
	    ofs << alpha[k] << " ";
	}
	ofs << "\n";
	// for (v = 0; v < num_vocab; v++){
	    // ofs << eta[v] << " ";
	// }
	// ofs << "\n";
    ofs.close();
}


void LDA::show_vlb(){
    calc_vlb();
    cout << "VLB: " << temp_vlb << endl;
    cout << "Improvement point: " << temp_vlb - old_vlb << endl;
}

void LDA::Normalize(vector<double> &vec, int &length){
	double sum = 0;
    for (int i=0; i < length; i++){
        sum += vec[i];
    }
    for (int i=0; i < length; i++){
        vec[i] = vec[i] / sum;
    }
}

void run_VB_LDA(int num_topic, int num_data, int threshold, int experiment);
