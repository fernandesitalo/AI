#include <bits/stdc++.h>
using namespace std;

typedef vector<int> vi;
typedef long double lf;

lf rnd(){ return (((lf) rand() / (RAND_MAX))) * (rand()%2 == 0? 1 : -1); } // rand entre 0 e 1

/*Uma taxa de aprendizada alta acelera o treino porem pode fazer com a rede passe do ponto ótimo, uma taxa baixa diminui a velocidade de treino mas aumenta a precisão da busca. */


// colocar template depois
class Perceptron{
	public:
//O Perceptron é um classificador linear. Isso que dizer que ele
// só ira lidar com problemas de classificação onde o conjunto 
// de dados seja linearmente separável
	vector<vi> amostras;
	vi saidas;
	lf taxa_aprendizado; // la dita qual a proporção em que a rede ira propagar o erro  e assim realizar a atualização dos pesos.
	int epocas;
	lf limiar;
	vector<lf> pesos;
	lf eps;
	lf pesoLimiar;
	const function<int(lf)> &funcao;
	
	public:
	Perceptron(vector<vi> amostras, vi saidas,auto funcao,lf taxa_aprendizado=0.1,int epocas=1000,lf limiar=1, lf eps=1e-6) : funcao(funcao){
		this->amostras = amostras;
		this->saidas = saidas;
		this->taxa_aprendizado = taxa_aprendizado;
		this->epocas = epocas;
		this->limiar = limiar;
		this->eps = eps;
	}
	
	int executa(const vi &entrada){
		lf somatorio = limiar * pesoLimiar;
		for(int i = 0 ; i < (int)entrada.size() ; ++i)
			somatorio += pesos[i] * entrada[i];
		return somatorio;
	}
	
	void treinar(){
		// inicializacao dos pesos
		pesos.clear();
		for(int i = 0 ; i < (int)amostras.size() ; ++i)
			pesos.push_back(rnd());
		pesoLimiar = rnd();
		
		// erro = saidaEsperada - saidaPerceptron
		lf erro = 1;
		while(fabs(erro) > eps){
			erro = 0;
			for(int i = 0 ; i < (int)amostras.size() ; ++i){
				
				int saidaPerceptron = executa(amostras[i]);
				
				erro += (saidas[i]-saidaPerceptron);
				
				if(saidas[i] != saidaPerceptron){ // .: realizar ajuste
					lf erroAtual = (saidas[i]-saidaPerceptron);
					for(int j = 0 ; j < (int)pesos.size() ; ++j)
						pesos[j] = pesos[j] + (taxa_aprendizado * erroAtual * amostras[i][j]);
					pesoLimiar = pesoLimiar + (taxa_aprendizado * erroAtual * limiar);
				}
			}
		}
		cerr << "OK - treinado\n";
	}
};


int main(){
	srand(time(NULL));
	vector<vi> entradas;
	entradas.push_back({0,0});
	entradas.push_back({0,1});
	entradas.push_back({1,0});
	entradas.push_back({1,1});
	vector<int> saidas = {0,1,1,1};
	
	auto funcao = [](lf val){ return val>=0;};
	
	Perceptron P(entradas,saidas,funcao);
	P.treinar();
	int a,b;
	while(1){
		cin >> a >> b;
		cout << P.executa({a,b}) << '\n';
	}
	
	return 0;
}
