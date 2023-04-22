#esqueleto de grid search
#se espera que los alumnos completen lo que falta para recorrer TODOS cuatro los hiperparametros 

#Ver vídeo optimización de hiperparámetros en un árbol de decisión, algoritmo CART
#Se debe optimizar los hiperparametros con 5-fold cross validation a través de rpart_params

rm( list=ls() )  #Borro todos los objetos
gc()   #Garbage Collection

require("data.table")
require("rpart")
require("parallel")

ksemillas  <- c(114689, 274177, 333679, 514229, 8545543) #reemplazar por las propias semillas

#------------------------------------------------------------------------------
#particionar agrega una columna llamada fold a un dataset que consiste en una particion estratificada segun agrupa
# particionar( data=dataset, division=c(70,30), agrupa=clase_ternaria, seed=semilla)   crea una particion 70, 30 

particionar  <- function( data,  division, agrupa="",  campo="fold", start=1, seed=NA )
{
  if( !is.na(seed) )   set.seed( seed )

  bloque  <- unlist( mapply(  function(x,y) { rep( y, x )} ,   division,  seq( from=start, length.out=length(division) )  ) )  

  data[ , (campo) :=  sample( rep( bloque, ceiling(.N/length(bloque))) )[1:.N],
          by= agrupa ]
}
#------------------------------------------------------------------------------

ArbolEstimarGanancia  <- function( semilla, param_basicos )
{
  #particiono estratificadamente el dataset
  particionar( dataset, division=c(7,3), agrupa="clase_ternaria", seed= semilla )

  #genero el modelo
  modelo  <- rpart("clase_ternaria ~ .",     #quiero predecir clase_ternaria a partir del resto
                   data= dataset[ fold==1],  #fold==1  es training,  el 70% de los datos
                   xval= 0,
                   control= param_basicos )  #aqui van los parametros del arbol

  #aplico el modelo a los datos de testing
  prediccion  <- predict( modelo,   #el modelo que genere recien
                          dataset[ fold==2],  #fold==2  es testing, el 30% de los datos
                          type= "prob") #type= "prob"  es que devuelva la probabilidad

  #prediccion es una matriz con TRES columnas, llamadas "BAJA+1", "BAJA+2"  y "CONTINUA"
  #cada columna es el vector de probabilidades 


  #calculo la ganancia en testing  qu es fold==2
  ganancia_test  <- dataset[ fold==2, 
                             sum( ifelse( prediccion[, "BAJA+2"]  >  0.025,
                                         ifelse( clase_ternaria=="BAJA+2", 117000, -3000 ),
                                         0 ) )]

  #escalo la ganancia como si fuera todo el dataset
  ganancia_test_normalizada  <-  ganancia_test / 0.3

  return( ganancia_test_normalizada )
}
#------------------------------------------------------------------------------

ArbolesMontecarlo  <- function( semillas, param_basicos )
{
  #la funcion mcmapply  llama a la funcion ArbolEstimarGanancia  tantas veces como valores tenga el vector  ksemillas
  ganancias  <- mcmapply( ArbolEstimarGanancia, 
                          semillas,   #paso el vector de semillas, que debe ser el primer parametro de la funcion ArbolEstimarGanancia
                          MoreArgs= list( param_basicos),  #aqui paso el segundo parametro
                          SIMPLIFY= FALSE,
                          mc.cores= 1 )  #se puede subir a 5 si posee Linux o Mac OS

  ganancia_promedio  <- mean( unlist(ganancias) )

  return( ganancia_promedio )
}
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

#Aqui se debe poner la carpeta de la computadora local
setwd("C:/Users/jball/OneDrive/Documentos/Labo/")   #Establezco el Working Directory
#cargo los datos

#cargo los datos
dataset  <- fread("./datasets/dataset_pequeno.csv")

#trabajo solo con los datos con clase, es decir 202107
dataset  <- dataset[ clase_ternaria!= "" ]

#genero el archivo para Kaggle
#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/HT2020/", showWarnings = FALSE )
archivo_salida  <- "./exp/HT2020/gridsearch.txt"

#Escribo los titulos al archivo donde van a quedar los resultados
#atencion que si ya existe el archivo, esta instruccion LO SOBREESCRIBE, y lo que estaba antes se pierde
#la forma que no suceda lo anterior es con append=TRUE
cat( file=archivo_salida,
     sep= "",
     "max_depth", "\t",
     "min_split", "\t",
     "ganancia_promedio", "\n")


#itero por los loops anidados para cada hiperparametro
#Entender que minbucket debe ser:  2 <= 2*minbucket <= minsplit <= #dataset
#Entender que max_depth debe ser:   2 <= max_depth <= 30
#Entender que -1 <= cp <= 0.1  (cp está entre -1 y 0.1)

for( vcp  in  c( -0.6, -0.5, -0.4)  )
{
for( vmax_depth  in  c( 4, 6, 8, 10, 12, 14 )  )
{
for( vmin_split  in  c( 800, 600, 400, 200, 100, 50, 20, 10)  )
{
for( vmin_bucket  in  c( vmin_split/2, 20, 10, 5) )
{

  #notar como se agrega
  param_basicos  <- list( "cp"=        vcp,       #complejidad minima
                          "minsplit"=  vmin_split,  #minima cantidad de registros en un nodo para hacer el split
                          "minbucket"= vmin_bucket,          #minima cantidad de registros en una hoja  (prepoda)
                          "maxdepth"=  vmax_depth ) #profundidad máxima del arbol

  #Un solo llamado, con la semilla 17
  ganancia_promedio  <- ArbolesMontecarlo( ksemillas,  param_basicos )

  #escribo los resultados al archivo de salida
  cat(  file=archivo_salida,
        append= TRUE,
        sep= "",
        vmax_depth, "\t",
        vmin_split, "\t",
        ganancia_promedio, "\n"  )

}
}
}
}


#El Grid Search es algo que no se utiliza en la vida real , por más que hay penosos artículos de TowardsDataScience y Medium que hablan de él
#Estamos viendo Grid Search simplemente como un paso pedagógico intermedio, con los for anidados, donde nos es muy importante generar un archivo con todas las salidas , donde el secreto va a ser ver ese archivo de salida como un dataset
#Ese "dataset" ,. al mirarlo con ingenio, nos va a dar lugar a la "Optimización Bayesiana* , tema que veremos en la segunda mitad de la Clase 02, el proximo martes, y es totalmente superador al Grid Search.
#Para hacer Bayesian Optimization utilizaremos dos librerías : DiceKriging* y mlrMBO**
#incluso habrán fascículos de zero2hero que hablaran de los fundamentos de la Bayesian Optimization