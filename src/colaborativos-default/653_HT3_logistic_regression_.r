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
require ("caretensemble")
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

PARAM$glm_crossvalidation_folds  <- 5L  #En caso que se haga cross validation, se usa esta cantidad de folds

PARAM$glm_semilla  <- 114689   #cambiar por su propia semilla



#Hiperparametros FIJOS de regresión logística
PARAM$glm_basicos <- list( 
   family= "binomial",
   seed=  PARAM$glm_semilla
)

#Aqui se cargan los hiperparametros que se pueden optimizar en la Bayesian Optimization, para el caso no se hizo BO

PARAM$glm_extras <- makeParamSet( 
         makeNumericParam("cost",    lower=    0.01, upper=     10),
         makeNumericParam("penaly", lower=    0.001, upper=     0.1),
         makeIntegerParam("maxit", lower=  50L,   upper= 200L),
         makeIntegerParam("k", lower = PARAM$glm_crossvalidation_folds, upper=10L)
)

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

dataset[ , azar :=  NULL ]

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

  set.seed( PARAM$glm_semilla )
  modelo_glm_train  <- train(clase_ternaria ~ numero_de_cliente + foto_mes,
                            data= dataset,
                            method = "glm",
                            trControl= trainControl(
                                method = "cv",
                                number = 6,
                                summaryFunction = twoClassSummary,
                                classProbs = TRUE,
                                verboseIter = TRUE
                                )
                            )
  modelo_glm_train
  modelo_glm_test <- dataset[-modelo_glm_train]
  modelo_glm_test

  table(modelo_glm_train$clase_ternaria)  %>% prop.table()
  nrow(modelo_glm_train)
  table(modelo_glm_test$clase_ternaria)  %>% prop.table()
  nrow(modelo_glm_test)

  prediccion<-predict(modelo_glm_test, type="response")
  prediccion_clase <- ifelse(p>0.025, 1, 0)
  confusionMatrix(prediccion_clase, modelo_glm_test$clase_ternaria)
  colAUC(p,modelo_glm_test$clase_ternaria, plotROC)

    
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
