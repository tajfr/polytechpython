#TAJ FAROUI
#TP MONTECARLO

#IMPORTS

import random
import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
#import plotly.plotly as py

#---------------------------------TP 1--------------------------------------

#FONCTIONS

random.seed() #ON FIXE LA GRAINE OU NON

def gen(n):
    i=0
    while i<n:
        print(random.random())
        i=i+1
    return

def bin():
    x=random.random()
    if x<0.5:
        return 1
    else:
        return 0

def histo(n,h):
    i=0
    j=0
    z=math.floor(1/h)
    A=[]
    H=[]
    
    #COMPTEUR DE VALEURS INITIALISé A 0 POUR CHAQUE INTERVALLE
    while i<z:
        H.append(0)
        i=i+1

    #GENERATION DES NOMBRES ALEATOIRES ENTRE 0 ET 1
    #ON LES STOCKE DANS UNE LISTE ET ON LES COMPTE PAR INTERVALLE
    
    while j<n:
        x=random.random()

        A.append(x)

        p=math.floor(x/h)
        H[p]=H[p]+1

        j=j+1

    plt.hist(A, bins=z,edgecolor = 'black',normed=1)
    plt.xlabel('Valeurs')
    plt.ylabel('Quantités')
    plt.title('Histogramme des tirages entre 0 et 1')
    plt.show()
    return H


def loiexpo(lamb,n):
    i=0
    j=0
    H=[]

    while j<n:
        x=random.random()
        p=-math.log(x)/lamb
        H.append(p)

        j=j+1

    E=sum(H)/n

    VAR=0
    a=0

    for i in range(0, len(H)):
        a=a+H[i]*H[i]

    VAR=(n/(n-1))*((a/n)-E*E)

    borneinf = round(E - (1.96*math.sqrt(VAR)/math.sqrt(n)),7) #4 chiffres apres la virgule
    bornesup = round(E + (1.96*math.sqrt(VAR)/math.sqrt(n)),7) #suffisent

    inintervalle = False

    if ((1/lamb) >= borneinf):     
        if ((1/lamb)<= bornesup):
            inintervalle = True

    print("Esperance : ",E)
    print("Variance : ",VAR)
    print("Intervalle de confiance pour alpha=95% : [",borneinf," ; ",bornesup,"]")
    print("Esperance dans l'intervalle : ",inintervalle)
    
    t = np.arange(0.0, 1, 0.01)
    s = lamb*np.exp(-lamb*t)
    
    plt.plot(t, s,'r')
    plt.hist(H, bins=50,edgecolor = 'black', normed=lamb)
    plt.xlabel('temps t')
    plt.ylabel('y(t)')
    plt.title('Comparaison entre la loi exponentielle et notre tirage')
    plt.show()
        
    return

def bern(p,n):
    j=0
    H=[0,0]

    while j<n:
        x=random.random()
        if x<=p:
            H[0]=H[0]+1
        else:
            H[1]=H[1]+1
        j=j+1

    phat=H[0]/n
    return phat

def pchapeau (p,n,m):
    H=[]
    i=0
    while i<m:
        phat=bern(p,n+50*i) #convergence du phat vers p selon l'augmentation du nombre de tirages
        H.append(phat)
        i=i+1

    #on affiche tous nos phat
    plt.plot(H,'g^')

    #on trace la fonction constante egale a p comme base de comparaison
    plt.axhline(y=p,xmin=0,xmax=m,c="red",linewidth=0.5,zorder=0)
    plt.ylabel('p = 0.45')
    plt.title("Observation de la convergence de phat vers p")
    plt.show()
    return

def boxmuller (n):
    i=0
    H=[]

    while i<n:
       x=random.random()
       y=random.random()

       g1=math.sqrt(-2*math.log(x))*math.cos(2*math.pi*y)
       g2=math.sqrt(-2*math.log(x))*math.sin(2*math.pi*y)

       H.append(g1)
       H.append(g2)

       i=i+1

    #t = np.arange(-5, 5, 0.01)
    #s = (1/math.sqrt(2*math.pi))*np.exp(-(0.5)*t*t)
    
    #plt.plot(t, s,'r')
    #plt.hist(H, bins=50,edgecolor = 'black',normed=0.4)
    #plt.xlabel('Valeurs')
    #plt.ylabel('Distribution')
    #plt.title('Comparaison entre la loi normale et la Méthode de Box-Muller')
    #plt.show()

    return

def rejetpolaire(n,mu,sigma):
    i=0
    H=[]

    while i<n:
        x=random.uniform(-1, 1)
        y=random.uniform(-1, 1)

        S=x*x+y*y

        if math.sqrt(S)<=1: #Rejet des points en dehors du cercle

            k1=(math.sqrt(-2*math.log(S)/S)*x)*sigma+mu
            k2=(math.sqrt(-2*math.log(S)/S)*y)*sigma+mu

            H.append(k1)
            H.append(k2)

        i=i+1        

    t = np.arange(mu-10, mu+10, 0.01)
    s = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-(1/2)*((t-mu)*(t-mu))/(sigma*sigma))
    
    
    plt.plot(t, s,'r')
    plt.hist(H, bins=50,edgecolor = 'black',normed=0.4)
    plt.xlabel('Valeurs')
    plt.ylabel('Distribution')
    plt.title('Comparaison entre la loi normale et la Méthode de Marsiglia')
    plt.show()

    return

def A(n,p,mu1,sigma1,mu2,sigma2):
    i=0
    H=[]
    E=0
    VAR=0

    while i<n:
        b=bern(p,1)

        if b==1: #BOX-MULLER avec mu1 et sigma1
            x=random.random()
            y=random.random()

            g1=math.sqrt(-2*math.log(x))*math.cos(2*math.pi*y)
            g2=math.sqrt(-2*math.log(x))*math.sin(2*math.pi*y)

            g1=g1*sigma1+mu1
            g2=g2*sigma1+mu1

            H.append(g1)
            H.append(g2)

            E=E+g1+g2 #on construit l'esperance au fur et a mesure

            VAR=VAR+g1*g1+g2*g2 #idem pour la variance
            
        else: #REJET POLAIRE avec mu2 et sigma2
            x=random.uniform(-1, 1)
            y=random.uniform(-1, 1)

            S=x*x+y*y

            if math.sqrt(S)<=1: #Rejet des points en dehors du cercle unité

                k1=(math.sqrt(-2*math.log(S)/S)*x)*sigma2+mu2
                k2=(math.sqrt(-2*math.log(S)/S)*y)*sigma2+mu2

                H.append(k1)
                H.append(k2)

                E=E+k1+k2 #comme ci-dessus

                VAR=VAR+k1*k1+k2*k2
        
        i=i+1

    E=E/n
    VAR=(n/(n-1))*((VAR/n)-E*E)

    borneinf = round(E - (1.96*math.sqrt(VAR)/math.sqrt(n)),4)
    bornesup = round(E + (1.96*math.sqrt(VAR)/math.sqrt(n)),4)

    #print("Esperance : ",E)
    #print("Variance : ",VAR)
    #print("Intervalle de confiance pour alpha=95% : [",borneinf,";",bornesup,"]")
    
    #mu=p*mu1+(1-p)*mu2 #LA SOURCE DE MON PROBLEME

    #sigma=p*sigma1+(1-p)*sigma2
    #t = np.arange(mu-10, mu+10, 0.01)
    #s = (1/(sigma*math.sqrt(2*math.pi)))*np.exp(-(1/2)*((t-mu)*(t-mu))/(sigma*sigma))
    
    #plt.plot(t, s,'r')    
    #plt.hist(H, bins=50,edgecolor = 'black',normed=0.4)
    #plt.xlabel('Valeurs')
    #plt.ylabel('Distribution')
    #plt.title('Représentation graphique de la loi A')
    #plt.show()
    
    return E

def vitconvA():
    i=1
    N=2500
    H=[]
    J=[]
    while i<50:
        H.append(i-1)
        J.append(A(N*i,0.6,1,2,10,2))
        i=i+1

    plt.plot(H,J,'g^')
    plt.show()
        
    return


#---------------------------------------------------------------------------
        
#PHASE DE TEST


#---------------------------------EXO 0-------------------------------------

#gen(7)
#print(histo(10000,0.1))

#---------------------------------EXO 1-------------------------------------

#print(loiexpo(7,15000))

#---------------------------------EXO 2-------------------------------------

#print(bern(0.45,1200))
#start_time = time.time()
#pchapeau (0.45,1200,1000)
#print("Temps d'execution convergence : %s secondes" % (time.time() - start_time))

#---------------------------------EXO 3-------------------------------------

#start_time = time.time()
#boxmuller(50000)
#print("Temps d'execution Box-Muller : %s secondes" % (time.time() - start_time))

#start_time = time.time()
#rejetpolaire(20000,1,3)
#print("Temps d'execution Marsiglia : %s secondes" % (time.time() - start_time))

#---------------------------------EXO 4 et 5-------------------------------------
#print(A(10000,0.6,1,2,10,2))
#vitconvA()

#---------------------------------TP 2--------------------------------------

def Euler(x0,n,NMC,K,delta):
    
    A=0
    
    A0=1
    A1=-5
    A2=2
    A3=3
    A4=5
    
    i=0
    j=0
    p=0
    
    x=x0
    
    while j<NMC: #NMC tirages pour l'obtention d'une esperance adequate
        while i<n: #schema d'euler
        
            g=np.random.normal(0,1)
            #EULER
            x=x+A0*(A1-x)*delta+((A2+A3*x*x)/(A4+A3*x*x))*math.sqrt(delta)*g
            #on ecrase la derniere valeur a chaque fois, seule la derniere nous interesse
            i=i+1

        p=p+max((K-x),0)
        A=A+max((K-x),0)*max((K-x),0)
        
        j=j+1


    E=p/NMC
    VAR=(NMC/(NMC-1))*((A/NMC)-E*E)

    #borneinf=round(E - (1.96*math.sqrt(VAR)/math.sqrt(NMC)),7)
    #bornesup=round(E + (1.96*math.sqrt(VAR)/math.sqrt(NMC)),7)
    
    #print("P(T,x) = ",E)
    #print("Intervalle de confiance à 95% = [",borneinf," ; ",bornesup,"]")

    return E

def Romberg(x0,n,NMC,K,delta):

    VAR1=0
    VAR2=0
    
    A0=1
    A1=-5
    A2=2
    A3=3
    A4=5

    i=0
    j=0

    P=0
    P2=0
    p1=0
    p2=0
    
    x=x0
    y=x0

    T=1;
    
    while j<NMC: 
        while i<n:
            T=1-T;

            if T==1:
                g2=np.random.normal(0,1)
                x=x+A0*(A1-x)*delta+((A2+A3*x*x)/(A4+A3*x*x))*math.sqrt(delta)*g2
                y=y+A0*(A1-y)*2*delta+((A2+A3*y*y)/(A4+A3*y*y))*math.sqrt(delta)*(g1+g2)
            else:
                g1=np.random.normal(0,1)
                x=x+A0*(A1-x)*delta+((A2+A3*x*x)/(A4+A3*x*x))*math.sqrt(delta)*g1
            
            i=i+1

        #p1=p1+max((K-x),0)
        #p2=p2+max((K-y),0)

        P=P+2*max((K-x),0)-max((K-y),0)
        P2=P2+(2*max((K-x),0)-max((K-y),0))*(2*max((K-x),0)-max((K-y),0))
        
        j=j+1

    #E1=p1/NMC
    #E2=p2/NMC

    #E=-E2+2*E1 #MASTER

    E=P/NMC
    
    VAR=(NMC/(NMC-1))*(P2/NMC - E*E)
    #print("P2=",P2)
    #print("E=",E)
    #print("VAR=",VAR)
    

    #borneinf=E - (1.96*math.sqrt(VAR)/math.sqrt(NMC))
    #bornesup=E + (1.96*math.sqrt(VAR)/math.sqrt(NMC))

    #print("P(T,x) = ",E)
    
    return E

def evol():
    
    #delta=0.001

    i=10
    H=[]
    J=[]
    K=[]
    
    while i>3:
        H.append(10**(-i))
        
        E1=Euler(1,10000,10000,50,2*10**(-i))
        E2=Romberg(1,10000,10000,50,10**(-i))
        
        J.append(E1)
        K.append(E2)

        #delta=delta/5
        i=i-0.2

    eu, = plt.plot(H,J)
    rom, = plt.plot(H,K)
    plt.legend([eu, rom], ['Euler', 'Romberg'])
    plt.xlabel('delta')
    plt.ylabel('Prix du Call obtenu par nos deux méthodes')
    plt.show()
    
    return

def evol1():
    
    #delta=0.001

    i=10
    H=[]
    J=[]
    K=[]
    
    while i>3:
        H.append(10**(-i))
        
        E1=Euler(1,10000,10000,50,2*10**(-i))
       
        
        J.append(E1)
        

        #delta=delta/5
        i=i-0.2

    plt.plot(H,J)
    plt.xlabel('delta')
    plt.ylabel('Prix du Call')
    plt.show()
    
    return


#---------------------TEST--------------------------

#Euler(1,1000,10000,5000,0.1)
print(evol1())

#---------------------------------TP 3--------------------------------------

def CallBS(S0,K,r,T,sigma):

    i=0
    S=0
    E=0
    VAR=0
    n=10000

    while i<n:
        g=np.random.normal(0,1)
        S=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g)

        #E=E+max((S-K),0)*math.exp(-r*T) #facteur d'actualisation ??
        E=E+max((K-S),0)*math.exp(-r*T)+S0-K*math.exp(-r*T) #parite call put
        #VAR=VAR+max((S-K),0)*math.exp(-r*T)*max((S-K),0)*math.exp(-r*T)
        VAR=VAR+(max((K-S),0)*math.exp(-r*T)+S0-K*math.exp(-r*T))*(max((K-S),0)*math.exp(-r*T)+S0-K*math.exp(-r*T))
        
        i=i+1

    
    E=E/n
    VAR=(n/(n-1))*((VAR/n)-E*E)

    borneinf=round(E - (1.96*math.sqrt(VAR)/math.sqrt(n)),2)
    bornesup=round(E + (1.96*math.sqrt(VAR)/math.sqrt(n)),2)

    print("Prix du Call Européen = ",E)
    print("Variance = ",VAR)
    print("Intervalle de confiance à 95% = [",borneinf," ; ",bornesup,"]")

    return VAR

def partie2(S0,r,K,T,sigma,lamb,n):

    i=0
    Xlamb=0
    ANMC=0
    BNMC=0
    VAR=0
    VARX=0
    EX=0

    while i<n:

        g=np.random.normal(0,1)
        ST=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g)
        
        X=math.exp(-r*T)*max((ST-K),0)
        Y=S0-K*math.exp(-r*T)+math.exp(-r*T)*max(K-ST,0)
        Z=X-Y

        ANMC=ANMC+Z*Z
        BNMC=BNMC+X*Z

        EX=EX+X
        VARX=VARX+X*X

        Xlamb=Xlamb+X-lamb*Z
        VAR=VAR+(X-lamb*Z)*(X-lamb*Z)

        i=i+1

    EX=EX/n

    XLAMB=Xlamb/n
    ANMC=ANMC/n
    BNMC=BNMC/n
    lambNMC=BNMC/ANMC
    VAR=(n/(n-1))*(VAR/n - XLAMB*XLAMB)
    VARX=(n/(n-1))*(VARX/n - EX*EX)

    #print("Xlambda=",XLAMB)
    #print("ANMC=",ANMC)
    #print("BNMC=",BNMC)
    #print("lambdaNMC=",lambNMC)
    #print("VAR(X)=",VARX)
    #print("VAR(X^lambda)=",VAR)
    
    return VAR

def plotPartie2lambda():

    l=2
    H=[]
    J=[]

    while l>-0.5:

        H.append(l)
        J.append(partie2(80,0.05,80,1,0.3,l,10000))

        l=l-0.1

    plt.plot(H,J)
    plt.xlabel('Valeur de lambda')
    plt.ylabel('Variance')
    plt.show()

    return

def plotPartie2m():

    n=1000
    H=[]
    J=[]

    while n<100000:

        H.append(n)
        J.append(partie2(80,0.05,80,1,0.3,0.67,n))

        n=n+1000

    plt.plot(H,J,'g^')
    plt.xlabel('Nombre de tirages NMC')
    plt.ylabel('m-hat-NMC')
    plt.show()

    return

def partie3_1(a):

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<10000:
    
        g=np.random.normal(0,1)

        E1=E1+math.exp(g)+1
        E2=E2+(math.exp(g+a)+1)*math.exp(-a*g-a*a/2)

        VAR1=VAR1+(math.exp(g)+1)*(math.exp(g)+1)
        VAR2=VAR2+((math.exp(g+a)+1)*math.exp(-a*g-a*a/2))*((math.exp(g+a)+1)*math.exp(-a*g-a*a/2))
            
        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    print("VAR(G) = ",VAR1)
    print("VAR(H) = ",VAR2)

    return



def partie3_2(S0,r,K,T,sigma,a):

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<n:
        g=np.random.normal(0,1)
        S1=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g)
        S2=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(a+g))

        E1=E1+max((S1-K),0)*math.exp(-r*T)
        E2=E2+max((S2-K),0)*math.exp(-r*T)*math.exp(-a*g-a*a/2)

        VAR1=VAR1+max((S1-K),0)*math.exp(-r*T)*max((S1-K),0)*math.exp(-r*T)
        VAR2=VAR2+max((S2-K),0)*math.exp(-r*T)*math.exp(-a*g-a*a/2)*max((S2-K),0)*math.exp(-r*T)*math.exp(-a*g-a*a/2)

        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    #print("VAR_G = ",round(VAR1,2))
    #print("VAR_(a+G)= ",round(VAR2,2))
    
    return VAR2

def plotPartie3():

    a=-0.5
    H=[]
    J=[]

    while a<3:

        H.append(a)
        J.append(partie3_2(80,0.05,80,1,0.3,a))

        a=a+0.1

    plt.plot(H,J)
    plt.xlabel('Paramètre a')
    plt.ylabel('Variance')
    plt.show()

    return

def partie4_1():

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<10000:
    
        g=np.random.normal(0,1)

        E1=E1+math.exp(g)+1
        E2=E2+((math.exp(g)+1)+(math.exp(-g)+1))/2

        VAR1=VAR1+(math.exp(g)+1)*(math.exp(g)+1)
        VAR2=VAR2+((math.exp(g)+1)+(math.exp(-g)+1))*((math.exp(g)+1)+(math.exp(-g)+1))/4
            
        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    print("VAR_normale = ",VAR1)
    print("VAR_antithe = ",VAR2)

    return

def partie4_2(S0,r,K,T,sigma):

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<n:
        g=np.random.normal(0,1)
        S1=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g)
        S2=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(-g))

        E1=E1+max((S1-K),0)*math.exp(-r*T)
        E2=E2+(max((S1-K),0)*math.exp(-r*T)+max((S2-K),0)*math.exp(-r*T))/2

        VAR1=VAR1+max((S1-K),0)*math.exp(-r*T)*max((S1-K),0)*math.exp(-r*T)
        VAR2=VAR2+(max((S1-K),0)*math.exp(-r*T)+max((S2-K),0)*math.exp(-r*T))*(max((S1-K),0)*math.exp(-r*T)+max((S2-K),0)*math.exp(-r*T))/4
        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    print("VAR_normale = ",VAR1)
    print("VAR_antithe = ",VAR2)
    
    return

def supermaster(S0,r,K,T,sigma,a):

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<n:
        g1=np.random.normal(0,1)
        
        S8=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g1)
        S1=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(a+g1))
        S2=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(a-g1))

        ZZ=((max((K-S1),0)*math.exp(-r*T)+S0-K*math.exp(-r*T))*math.exp(-a*g1-a*a/2)+(max((K-S2),0)*math.exp(-r*T)+S0-K*math.exp(-r*T))*math.exp(a*g1-a*a/2))/2

        E1=E1+max((S8-K),0)*math.exp(-r*T)
        E2=E2+ZZ
        
        VAR1=VAR1+(max((S8-K),0)*math.exp(-r*T))*(max((S8-K),0)*math.exp(-r*T))
        VAR2=VAR2+ZZ*ZZ
        
        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    print("VAR1 = ",VAR1)
    print("VAR2 = ",VAR2)

    return

def supermaster_lamb(S0,r,K,T,sigma,a,lamb):

    i=0
    n=10000
    E1=0
    E2=0
    VAR1=0
    VAR2=0

    while i<n:
        g1=np.random.normal(0,1)
        
        S8=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*g1)
        S1=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(a+g1))
        S2=S0*math.exp((r-sigma*sigma/2)*T+sigma*math.sqrt(T)*(a-g1))

        X1=max((S1-K),0)*math.exp(-r*T)
        Y1=max((K-S1),0)*math.exp(-r*T)+S0-K*math.exp(-r*T)
        Z1=X1-Y1
        XL1=X1-lamb*Z1

        X2=max((S2-K),0)*math.exp(-r*T)
        Y2=max((K-S2),0)*math.exp(-r*T)+S0-K*math.exp(-r*T)
        Z2=X2-Y2
        XL2=X2-lamb*Z2

        ZZ=(XL1*math.exp(-a*g1-a*a/2)+XL2*math.exp(a*g1-a*a/2))/2

        E1=E1+max((S8-K),0)*math.exp(-r*T)
        E2=E2+ZZ
        
        VAR1=VAR1+(max((S8-K),0)*math.exp(-r*T))*(max((S8-K),0)*math.exp(-r*T))
        VAR2=VAR2+ZZ*ZZ
        
        i=i+1

    E1=E1/n
    E2=E2/n

    VAR1=(n/(n-1))*(VAR1/n-E1*E1)
    VAR2=(n/(n-1))*(VAR2/n-E2*E2)

    #print("VAR1 = ",VAR1)
    #print("VAR2 = ",VAR2)

    return VAR2

def tempsDeCalcul():

    start_time = time.time()
    A=CallBS(80,80,0.05,1,0.3)
    print("Temps d'execution Call Normal : %s secondes" % round((time.time() - start_time),3))
    print("Produit VAR*Temps_execution = ",round(A*(time.time() - start_time),3))

    start_time = time.time()
    B=supermaster_lamb(80,0.05,80,1,0.3,-0.29,1.11)
    print("Temps d'execution Variance Réduite : %s secondes" % round((time.time() - start_time),3))
    print("Produit VAR*Temps_execution = ",round(B*(time.time() - start_time),3))

    return

#---------------------TEST--------------------------

#CallBS(80,80,0.05,1,0.3)
#partie2(80,0.05,80,1,0.3,0.67,10000)
#plotPartie2m()

#partie3_1(1)
#partie3_2(80,0.05,80,1,0.3,1)
#plotPartie3()

#partie4_1()
#partie4_2(80,0.05,80,1,0.3)

#supermaster(80,0.05,80,1,0.3,-0.5)
#supermaster_lamb(80,0.05,80,1,0.3,-0.29,1.11)

#tempsDeCalcul()
