%Dados de entrada
format long
ar1= 'buildings_tex';
ar2='pavedroads_tex';

%%
%dados das distribuicoes que geraram o data
%entrada de dados
%matriz de dados das features das classes 
DIRE='DATA2009/';
arivo1= sprintf('%s%s.txt', DIRE, ar1);
arivo2= sprintf('%s%s.txt', DIRE, ar2);
disp(arivo2)

E1 = load(arivo1, '-ascii');
E2 = load(arivo2, '-ascii');

%normalizacao dos valores de entrada
%normalizacao dos dados de entrada da classe 1 e 2

maxi1 = max(E1);
mini1 = min(E1);

maxi2 = max(E2);
mini2 = min(E2);
minimo=min(mini1, mini2);
maximo=max(maxi1, maxi2);

faixa = maximo-minimo; %faixa de variacao

%normalizacao classe 1
[ne1,vars1] = size(E1);
NE1 = zeros(ne1,vars1);


for i=1:ne1
    for j=1:vars1
          if (E1(i,j)~=0)
            NE1(i,j) = (E1(i,j)- minimo(j) )/faixa(j);
          else
            NE1(i,j) = 0;  
          end
    end
end
%selecao dos mesmo conjunto de valores 
%para estimativa de JSD para classe 1
halfne1 = floor(ne1/2);
NNE1 = NE1(1:halfne1,:);

% hstograma 1
bins=100;
paso=1/bins;
HISTO=zeros(bins+1,vars1);
b=0;

for j=1:vars1
    for i=1:halfne1
          if (NNE1(i,j)~=0)
             x=NNE1(i,j);
             ponto=floor(x/paso)+1;
             HISTO(ponto,j)=HISTO(ponto,j)+1;
          else
             b = b+1;
          end
    end
end

SOMA=sum(HISTO);

%probabilidades dos descritores da classe 1
for j=1:vars1
    for i=1:bins+1
          if (SOMA(j)==0)
            Prob1(i,j) = 0;
          else
              %x=NE1(i,j);
              Prob1(i,j)=HISTO(i,j)/SOMA(j);
          end
    end
end

%normalizacao dos dados de entrada da classe 2
[ne2,vars1] = size(E2);
NE2 = zeros(ne2,vars1);

for i=1:ne2
    for j=1:vars1
          if (E2(i,j)~=0)
            NE2(i,j) = (E2(i,j)- minimo(j) )/faixa(j);
          else
            NE2(i,j) = 0;  
          end
    end
end


%selecao dos mesmo conjunto de valores 
%para estimativa de JSD para classe 1
halfne2 = floor(ne2/2);
NNE2 = NE2(1:halfne2,:);

% histograma 2
bins=100;
paso=1/bins;
HISTO1=zeros(bins+1,vars1);
b=0;

for j=1:vars1
    for i=1:halfne2
          if (NNE2(i,j)~=0)
             x=NNE2(i,j);
             ponto=floor(x/paso)+1;
             HISTO1(ponto,j)=HISTO1(ponto,j)+1;
          else
             b = b+1;
          end
    end
end

SOMA1=sum(HISTO1);
Prob2 = zeros(bins,vars1);

%probabilidades dos descritores da classe 2
for j=1:vars1
    for i=1:bins+1
          if (SOMA1(j)==0)
            Prob2(i,j) = 0;
          else
              %x=NE2(i,j);
              Prob2(i,j)=HISTO1(i,j)/SOMA1(j);
          end
    end
end
%%
%Separacao das amostras de controle e de verificacao
%Para classe 1
halfne1 = floor(ne1/2);
CNE1 = NE1(1:halfne1,:);
VNE1 = NE1((halfne1+1):ne1,:);

%Para classe 2
halfne2 = floor(ne2/2);
CNE2 = NE2(1:halfne2,:);
VNE2 = NE2((halfne2+1):ne2,:);

%%
%Calculo do vetor JSD
%calculo das divergencias KL

%vetor de probabilidade média
R = (Prob1+Prob2)/2;

%KL para Prob1 e R
num=bins+1;
DK1R=zeros(vars1, vars1);

for v1=1:vars1
    for v2=1:vars1
        % comparar
        dk=0;
        for i=1:num
            if (Prob1(i,v2)==0 || R(i,v1)==0) 
                a=1;
            else
                dk = dk +  Prob1(i,v2)*( log(Prob1(i,v2)/R(i,v1) ) );
                DK1R(v1,v2)=dk;
            end
        end
    end
end

for v1=1:vars1
    diag1(v1)=DK1R(v1,v1);
end

%KL para Prob2 e R
num=bins+1;
DK2R=zeros(vars1, vars1);

for v1=1:vars1
    for v2=1:vars1
        % comparar
        dk=0;
        for i=1:num
            if (Prob2(i,v2)==0 || R(i,v1)==0) 
                a=1;
            else
                dk = dk +  Prob2(i,v2)*( log(Prob2(i,v2)/R(i,v1) ) );
                DK2R(v1,v2)=dk;
            end
        end
    end
end

for v1=1:vars1
    diag2(v1)=DK1R(v1,v1);
end

%calculo JSD
JSDiv = (0.5*diag1)+(0.5*diag2);
JSDiv';

%%
%Colocando os JSD em ordem decrescente e guardando a posição
[JSDdes,pos] = sort(JSDiv, 'descend');
JSD_Fea = [JSDdes;pos]'
interm = zeros(vars1,4);

for j=1:vars1
%Calculo da distancia entre as amostras de controle e as classes
feature1c1 = CNE1(:,pos(j));
feature1c2 = CNE2(:,pos(j));
%medias e desvio-padrao da variavel (descritor)
mf1c1 = mean(feature1c1);
mf1c2 = mean(feature1c2);
stdf1c1 = std(feature1c1);
stdf1c2 = std(feature1c2);
%calculo das distancias
%distancias entre as amostras de verificacao das classes 1 e 2 em
%em relacao ao descritor
acertosC1 = 0;
errosC1 = 0;

for i=1:halfne1
    dc1_1(i) = abs((VNE1(i,pos(j))-mf1c1)/stdf1c1);
    dc2_1(i) = abs((VNE1(i,pos(j))-mf1c2)/stdf1c2);
    if dc1_1(i) < dc2_1(i)
        acertosC1 = acertosC1 + 1;
    else
        errosC1 = errosC1 +1;
    end
    interm(j,1) = acertosC1;
    interm(j,2) = errosC1;
end

acertosC2 = 0;
errosC2 = 0;

for i=1:halfne2
    dc1_2(i) = abs((VNE2(i,pos(j))-mf1c1)/stdf1c1);
    dc2_2(i) = abs((VNE2(i,pos(j))-mf1c2)/stdf1c2);
    if dc2_2(i) < dc1_2(i)
        acertosC2 = acertosC2 + 1;
    else
        errosC2 = errosC2 +1;
    end
    interm(j,3) = acertosC2;
    interm(j,4) = errosC2;
end

end
%%
%acuracia global
ag = zeros(1,vars1);


for i=1:vars1
    ag(i) = (interm(i,1)+interm(i,3))/sum(interm(i,:));
end

%identificando as features pelos seus indices
AG_Fea = [ag;pos]'

%%
%Comparacao entre melhores valores de JSD e AG
%Construir matrizes com os 5 melhores valores de JSD
%acompanhados dos valores de acurácia global e features
[n,m] = size(AG_Fea);
features = zeros(5,4);
g=0;

for i=1:5
    for j=1:n
        if JSD_Fea(i,2) == AG_Fea(j,2)
           features(i,1)= JSD_Fea(i,1);
           features(i,2)= AG_Fea(j,1);
           features(i,3)= AG_Fea(j,2);
        else
            g = g+1;
        end
    end
end

features(:,1)
features(:,2)
features(:,3)

%%
%Separacao das amostras de controle e de verificacao
% %Para classe 1
% halfne1 = floor(ne1/2);
% EE1 = E1(1:halfne1,:);
% 
% %Para classe 2
% halfne2 = floor(ne2/2);
% EE2 = E2(1:halfne2,:);
A = 0;
B = 0;

%medias e desvios-padrao
for j=1:vars1
%Calculo da distancia entre as amostras de controle e as classes
feature1c1 = E1(:,j);
feature1c2 = E2(:,j);
%medias e desvio-padrao da variavel (descritor)
mf1c1(j) = mean(feature1c1);
mf1c2(j) = mean(feature1c2);
stdf1c1(j) = std(feature1c1);
stdf1c2(j) = std(feature1c2);
%vetor dos valores de separação entre as classes:
    if mf1c1(j)>mf1c2(j)
        A = mf1c1(j)-stdf1c1(j);
        B = mf1c2(j)+stdf1c2(j);
        vetorsep(j) = (A+B)/2;
    else
        A = mf1c1(j)+stdf1c1(j);
        B = mf1c2(j)-stdf1c2(j);
        vetorsep(j) = (A+B)/2;
    end
end

feanum = [1:1:vars1];
u = [vetorsep;feanum];
limiar = u';
p=0;
%escolhendo os cinco limiares
% 
for i=1:vars1
    for j=1:5
        if limiar(i,2) == features(j,3)
           features(j,4) = limiar(i,1);
        else
            p = p+1;
        end
    end
end
features(:,4)