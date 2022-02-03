Embedding C inside Fortran with indirect indexing of sparse structs
===================================================================

#Base: Fortran support for C language
(Testato con compilazione in file oggetto per i C, poi linking insieme per il fortran)
[triv.	Tutti gli accessi nel codice C seguiranno la sintassi del C, (e.g. in C vFromF90[0] == vFromF90(1) in Fort.)]

#Refactoring needed for SpXXX operations
funzioni su strutture dati sparse richiedono shifting nell'uso di accessi a vettore 
con spiazzamento indiretto (tramite lista di indici
	-> uso di JA e IRP costruiti nel fortran richiede shifting di -1 ( - OFF\_F ) nel C da aggiungere
	   per tutti gli accessi mediante strutture di indici con semantica proveniente dal fortran.
	   
	  => Per questione di particità e leggibilità del codice, lo shifting è applicato a tutti gli indici
		 per accesso indiretto provenienti dal fortran, 
.		 __Ma nelle strutture e funzioni interne del C, gli indici saranno salvati con lo shifting applicato__
			Questo sia per semplicità e maggiore leggibilità del codice ( meglio avere meno macro in giro... )
			ma anche per questioni di efficienza dato che questo approccio dovrebbe portare a un overhead minore 
			in termini di istruzioni per la compatibilità C-Fortran.
.			__NB alcune componenti gestite mediante memcpy (colPartitionsAllocd)-> lasciati con indici originali, MA gestiti con shift__ 
.				__questi compomenti di strutture interne saranno quindi trattati come se fossero strutture arrivate dal Fortran__

__NB VICEVERSA nei dati in uscita dalle funzioni esportate del C verso il Fortran, si dovrà avere un ulteriore shifting per ritornare alla semantica Fortran__

#Approccio implementativo:
- realizzo lo shifting mediante operazioni di -1 agli'indici da shiftare utilizzando la macro: `OFF\_F` 
- dal mio pattern multi implementazione: git@github.com:andreadiiorio/C\_Compile\_Multi\_Implementation\_Automatically.git
  genero 2 versioni di ogni funzione interessata all'adattamento per compatibilità C-Fortran con aggiunta di 
  `CAT(funcName+\_,OFF\_F)` come nome della funzione che espanderà,
  mediante una coppia di `#include` e ridefinizione di `OFF\_F`, in una coppia di implementazioni con i valori di OFF\_F pari a 
	- 0 (uso solo nel C, inoltre il compilatore dovrebbe eliminare inutili operazioni di - 0 nel codice)
	- 1 (uso del codice C internamente a una applicazione Fortran)

