# Experimentos Colaborativos Default
# Hyperparameter Tuning  glm regresión logísitica
#ver https://rpubs.com/AdSan-R/Rlog_NB_ChurnBank

#Necesita para correr en Google Cloud
# 128 GB de memoria RAM
#   8 vCPU

# pensado para datasets con UNDERSAPLING de la clase mayoritaria

#limpio la memoria
rm( list= ls(all.names= TRUE) )  #remove all objects
gc( full= TRUE )                 #garbage collection

require("data.table")
require("rlist")
require("yaml")

#require("lightgbm")
require ("glmnet")
require ("caret")
#require ("caretensemble")
require ("ggplot2")

#paquetes necesarios para la Bayesian Optimization
#require("DiceKriging")
#require("mlrMBO")

#------------------------------------------------------------------------------
options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})
#------------------------------------------------------------------------------

#Parametros del script
PARAM  <- list()
PARAM$experimento <- "HT6530"

PARAM$exp_input  <- "TS6410"

PARAM$glm_crossvalidation_folds  <- 5  #En caso que se haga cross validation, se usa esta cantidad de folds

PARAM$glm_semilla1  <- 114689   #cambiar por su propia semilla
PARAM$glm_semilla2  <- 333679   #cambiar por su propia semilla
PARAM$glm_semilla3  <- 274177   #cambiar por su propia semilla
PARAM$glm_semilla4  <- 514229   #cambiar por su propia semilla
PARAM$glm_semilla5  <- 545543   #cambiar por su propia semilla


#Hiperparametros FIJOS de regresión logística
PARAM$glm_basicos <- list( 
   family= "binomial",
   seed=  PARAM$glm_semilla,
   k= PARAM$glm_crossvalidation_folds
)

#Aqui se cargan los hiperparametros que se pueden optimizar en la Bayesian Optimization, para el caso no se hizo BO

#PARAM$glm_extras <- makeParamSet( 
#        makeNumericParam("cost",    lower=    0.01, upper=     10),
#         makeNumericParam("penaly", lower=    0.001, upper=     0.1),
#         makeIntegerParam("maxit", lower=  50L,   upper= 200L),
#         makeIntegerParam("k", lower = PARAM$glm_crossvalidation_folds, upper=10L)
#)

# FIN Parametros del script

#------------------------------------------------------------------------------
# Espacio para funciones
#------------------------------------------------------------------------------

#Aqui empieza el programa
PARAM$stat$time_start  <- format(Sys.time(), "%Y%m%d %H%M%S")

setwd("~/buckets/b1/")

#cargo el dataset donde voy a entrenar
#esta en la carpeta del exp_input y siempre se llama  dataset_training.csv.gz
dataset_input  <- paste0( "./exp/", PARAM$exp_input, "/dataset_training.csv.gz" )
dataset  <- fread( dataset_input )

#dataset[ , azar :=  NULL ]

#Verificaciones
if( ! ("fold_train"    %in% colnames(dataset) ) ) stop("Error, el dataset no tiene el campo fold_train \n")
if( ! ("fold_validate" %in% colnames(dataset) ) ) stop("Error, el dataset no tiene el campo fold_validate \n")
if( ! ("fold_test"     %in% colnames(dataset) ) ) stop("Error, el dataset no tiene el campo fold_test  \n")
if( dataset[ fold_train==1, .N ] == 0 ) stop("Error, en el dataset no hay registros con fold_train==1 \n")

#creo la carpeta donde va el experimento
dir.create( paste0( "./exp/", PARAM$experimento, "/"), showWarnings = FALSE )
setwd(paste0( "./exp/", PARAM$experimento, "/"))   #Establezco el Working Directory DEL EXPERIMENTO

write_yaml( PARAM, file= "parametros.yml" )   #escribo parametros utilizados

cat( PARAM$exp_input,
     file= "TrainingStrategy.txt",
     append= FALSE )

#defino la clase binaria clase01
dataset[  , clase01 := ifelse( clase_ternaria=="CONTINUA", 0L, 1L ) ]


#los campos que se pueden utilizar para la prediccion
campos_buenos  <- setdiff( copy(colnames( dataset )), c( "clase01", "clase_ternaria", "fold_train", "fold_validate", "fold_test" ) )


#//*************//#//*************//
  #str(dataset)
  #dataset$clase_ternaria<- as.factor(dataset$clase_ternaria)
  dataset$clase_ternaria<- ifelse( dataset$clase_ternaria == "BAJA+2", 1.0000001, 
        ifelse( dataset$clase_ternaria == "BAJA+1", 1.0, 0.0) )
 

dtrain<-dataset[ fold_train==1]
dtest<-dataset[fold_test==1]
dtest2<-dataset[ !fold_train==1]

set.seed( PARAM$glm_semilla1)  
modelo_glm  <- glm(clase01~numero_de_cliente + foto_mes, data= dtrain,
                  family= "binomial"
  )
confint(object = modelo_glm, level = 0.95 )
  
modelo_glm2  <- glm(clase01~foto_mes, data= dtrain,
                     family= "binomial"
  )

  
plot(clase01 ~ foto_mes, dataset, col = "darkblue",
       main = "Modelo regresión logística",
       ylab = "P(bajas=1|foto_mes)",
       xlab = "foto_mes", pch = "I")
  
# type = "response" devuelve las predicciones en forma de probabilidad en lugar de en log_ODDs
curve(predict(modelo_glm2, data.frame(foto_mes = x), type = "response"),
        col = "firebrick", lwd = 2.5, add = TRUE)

# Predicciones de los nuevos puntos según el modelo. 
# Si se indica se.fit = TRUE se devuelve el error estándar de cada predicción
# junto con el valor de la predicción (fit).
#predicciones <- predict(modelo_glm2, data.frame(foto_mes=dataset$foto_mes), se.fit = TRUE)
#prediccion <- predict(modelo_glm2, list(foto_mes=dtest$foto_mes), type="response",se.fit = TRUE)
   prediccion1 <- predict(modelo_glm2, list(foto_mes=dtest$foto_mes), type="response",
                           se.fit = TRUE)
   
   prediccion_p<-prediccion1

   tb_prediccion1  <- dtest[ , list( numero_de_cliente, foto_mes, clase_ternaria ) ]
   
   tb_prediccion1[ , prob := prediccion_p$fit]

# También mediante la función logit se transforman los log_ODDs a probabilidades.
predicciones_logit <- exp(predicciones$fit) / (1 + exp(predicciones$fit))

#probabilidades <- predict(modelo_glm2, data.frame(foto_mes=dataset$foto_mes), type="response", se.fit = TRUE)
#prob_predic <- predict(modelo_glm2, data.matrix(dtest), type="response", se.fit = TRUE)
#predic <- predict(modelo_glm2, data.matrix(dtest[_, campos_buenos, with= FALSE ], type="response", se.fit = TRUE))
  

set.seed( PARAM$glm_semilla2)  
modelo_glm_s2  <- glm(clase01~foto_mes, data= dtrain,
                     family= "binomial"
  )

set.seed( PARAM$glm_semilla3)  
modelo_glm_s3  <- glm(clase01~foto_mes, data= dtrain,
                     family= "binomial"
  )

set.seed( PARAM$glm_semilla4)  
modelo_glm_s4  <- glm(clase01~foto_mes, data= dtrain,
                     family= "binomial"
  )

set.seed( PARAM$glm_semilla5)  
modelo_glm_s5  <- glm(clase01~foto_mes, data= dtrain,
                     family= "binomial"
  )


  prediccion2 <- predict(modelo_glm_s2, list(foto_mes=dtest$foto_mes), type="response",
                         se.fit = TRUE)
  prediccion_p<-prediccion2
  tb_prediccion2  <- dtest[ , list( numero_de_cliente, foto_mes, clase_ternaria ) ]
  tb_prediccion2[ , prob := prediccion_p$fit]

  prediccion3 <- predict(modelo_glm_s3, list(foto_mes=dtest$foto_mes), type="response",
                         se.fit = TRUE)
  prediccion_p<-prediccion3
  tb_prediccion3  <- dtest[ , list( numero_de_cliente, foto_mes, clase_ternaria ) ]
  tb_prediccion3[ , prob := prediccion_p$fit]

  prediccion4 <- predict(modelo_glm_s4, list(foto_mes=dtest$foto_mes), type="response",
                         se.fit = TRUE)
  prediccion_p<-prediccion4
  tb_prediccion4  <- dtest[ , list( numero_de_cliente, foto_mes, clase_ternaria ) ]
  tb_prediccion4[ , prob := prediccion_p$fit]

  prediccion5 <- predict(modelo_glm_s5, list(foto_mes=dtest$foto_mes), type="response",
                         se.fit = TRUE)
  prediccion_p<-prediccion5
  tb_prediccion5  <- dtest[ , list( numero_de_cliente, foto_mes, clase_ternaria ) ]
  tb_prediccion5[ , prob := prediccion_p$fit]



  #Guardar archivos
  
  nom_pred  <- paste0( "pred1_",
                       "regresion lineal",
                       ".csv"  )
  
  fwrite( tb_prediccion[ ,list(numero_de_cliente, foto_mes, prob, clase_ternaria)],
          file= nom_pred,
          sep= "\t" )
  
  nom_pred  <- paste0( "pred2_",
                       "regresion lineal",
                       ".csv"  )
  
  fwrite( tb_prediccion2[ ,list(numero_de_cliente, foto_mes, prob, clase_ternaria)],
          file= nom_pred,
          sep= "\t" )
   
  nom_pred  <- paste0( "pred3_",
                       "regresion lineal",
                       ".csv"  )
  
  fwrite( tb_prediccion3[ ,list(numero_de_cliente, foto_mes, prob, clase_ternaria)],
          file= nom_pred,
          sep= "\t" )
  
  nom_pred  <- paste0( "pred4_",
                       "regresion lineal",
                       ".csv"  )
  
  fwrite( tb_prediccion4[ ,list(numero_de_cliente, foto_mes, prob, clase_ternaria)],
          file= nom_pred,
          sep= "\t" )
  
  nom_pred  <- paste0( "pred5",
                       "regresion lineal",
                       ".csv"  )
  
  fwrite( tb_prediccion[ ,list(numero_de_cliente, foto_mes, prob, clase_ternaria)],
          file= nom_pred,
          sep= "\t" )


  
  
#  prediccion<-predict(modelo_glm_test, type="response")
#  prediccion_clase <- ifelse(p>0.025, 1, 0)
#  confusionMatrix(prediccion_clase, modelo_glm_test$clase_ternaria)
#  colAUC(p,modelo_glm_test$clase_ternaria, plotROC)

    
 #set.seed( PARAM$glm_semilla )
 # modelo_train  <- glm.train( data= dtrain,
 #                             param=  param_completo
 #                           )

#//*************//#//*************//



#//*************//#//*************//
#//**********//BORRAR//**********//

#la particion de train siempre va
dtrain  <-  glm.probs( data=    data.matrix( dataset[ fold_train==1, campos_buenos, with=FALSE] ),
                        label=   dataset[ fold_train==1, clase01 ],
                        weight=  dataset[ fold_train==1, ifelse( clase_ternaria == "BAJA+2", 1.0000001, 
                                                                 ifelse( clase_ternaria == "BAJA+1", 1.0, 1.0) )],
                        free_raw_data= FALSE
                      )


kvalidate  <- FALSE
ktest  <- FALSE
kcrossvalidation  <- TRUE

#Si hay que hacer validacion
if( dataset[ fold_train==0 & fold_test==0 & fold_validate==1, .N ] > 0 )
{
  kcrossvalidation  <- FALSE
  kvalidate  <- TRUE
  dvalidate  <- lgb.Dataset( data=  data.matrix( dataset[ fold_validate==1, campos_buenos, with=FALSE] ),
                             label= dataset[ fold_validate==1, clase01 ],
                             weight= dataset[ fold_validate==1, ifelse( clase_ternaria == "BAJA+2", 1.0000001, 
                                                                     ifelse( clase_ternaria == "BAJA+1", 1.0, 1.0) )],
                             free_raw_data= FALSE  )

}


#Si hay que hacer testing
if( dataset[ fold_train==0 & fold_validate==0 & fold_test==1, .N ] > 0 )
{
  ktest  <- TRUE
  campos_buenos_test  <- setdiff( copy(colnames( dataset )), c( "fold_train", "fold_validate", "fold_test" ) )
  dataset_test  <- dataset[ fold_test== 1, campos_buenos_test, with=FALSE ]
}



rm( dataset )
gc()



#//***********//BORRAR//***********//
#//*************//#//*************//

#------------------------------------------------------------------------------
PARAM$stat$time_end  <- format(Sys.time(), "%Y%m%d %H%M%S")
write_yaml( PARAM, file= "parametros.yml" )   #escribo parametros utilizados

#dejo la marca final
cat( format(Sys.time(), "%Y%m%d %H%M%S"),"\n",
     file= "zRend.txt",
     append= TRUE  )

#------------------------------------------------------------------------------
#suicidio,  elimina la maquina virtual directamente
# para no tener que esperar a que termine una Bayesian Optimization 
# sino Google me sigue facturando a pesar de no estar procesando nada
# Give them nothing, but take from them everything.

system( "sleep 10  && 
        export NAME=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/name -H 'Metadata-Flavor: Google') &&
        export ZONE=$(curl -X GET http://metadata.google.internal/computeMetadata/v1/instance/zone -H 'Metadata-Flavor: Google') &&
        gcloud --quiet compute instances delete $NAME --zone=$ZONE",
        wait=FALSE )
