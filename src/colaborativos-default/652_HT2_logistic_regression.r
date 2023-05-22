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

##require("lightgbm")
require ("glmnet")

#paquetes necesarios para la Bayesian Optimization
require("DiceKriging")
require("mlrMBO")

#------------------------------------------------------------------------------
options(error = function() { 
  traceback(20); 
  options(error = NULL); 
  stop("exiting after script error") 
})
#------------------------------------------------------------------------------

#Parametros del script
PARAM  <- list()
PARAM$experimento <- "HT06510"

PARAM$exp_input  <- "TS6410"

PARAM$glm_crossvalidation_folds  <- 5  #En caso que se haga cross validation, se usa esta cantidad de folds

PARAM$glm_semilla  <- 114689   #cambiar por su propia semilla



#Hiperparametros FIJOS de regresión logística
PARAM$glm_basicos <- list( 
   family= "binomial",
   seed=  PARAM$glm_semilla
)

#Aqui se cargan los hiperparametros que se optimizan en la Bayesian Optimization

PARAM$bo_glm <- makeParamSet( 
         makeNumericParam("cost",    lower=    0.01, upper=     10),
         makeNumericParam("penaly", lower=    0.001, upper=     0.1),
         makeIntegerParam("maxit", lower=  50L,   upper= 200L)
)

##si usted es ambicioso, y tiene paciencia, podria subir este valor a 100
PARAM$bo_iteraciones  <- 50  #iteraciones de la Optimizacion Bayesiana, un poco más de 12 horas de proceso

# FIN Parametros del script

#------------------------------------------------------------------------------

EstimarGanancia_glmCV  <- function( x )
{
  gc()
  GLOBAL_iteracion  <<- GLOBAL_iteracion + 1

  param_completo  <- c(PARAM$glm_basicos,  x )

  #param_completo$early_stopping_rounds  <- as.integer(200 + 4/param_completo$learning_rate )

  vcant_optima   <<- c()
  GLOBAL_arbol  <<- 0
  GLOBAL_gan_max  <<- -Inf
 
  set.seed( PARAM$glm_semilla )  
  modelocv  <- cv.glm( data= dtrain,
                       param=  param_completo,
                       k= PARAM$glm_crossvalidation_folds
                     )

  cat("\n" )


  desde  <- (modelocv$best_iter-1)*PARAM$glm_crossvalidation_folds + 1
  hasta  <- desde + PARAM$glm_crossvalidation_folds -1


  cant_corte   <-  as.integer( mean( vcant_optima[ desde:hasta ] ) * PARAM$glm_crossvalidation_folds  )

  ganancia  <- unlist(modelocv$record_evals$valid$ganancia$eval)[ modelocv$best_iter ]
  ganancia_normalizada  <- ganancia * PARAM$glm_crossvalidation_folds


  if( ktest==TRUE )
  {
    #debo recrear el modelo
    param_completo$early_stopping_rounds  <- NULL
    param_completo$num_iterations  <- modelocv$best_iter

    modelo  <- glm.train( data= dtrain,
                          param=  param_completo
                        )


    #aplico el modelo a testing y calculo la ganancia
    prediccion  <- predict( modelo, 
                            data.matrix( dataset_test[ , campos_buenos, with=FALSE]) )

    tbl  <- copy( dataset_test[ , list("gan" = ifelse(clase_ternaria=="BAJA+2", 117000, -3000 )) ] )

    tbl[ , prob := prediccion ]
    setorder( tbl, -prob )
    tbl[ , gan_acum :=  cumsum( gan ) ]
    tbl[ , gan_suavizada :=  frollmean( x=gan_acum, n=2001, align="center", na.rm=TRUE, hasNA= TRUE )  ]


    #Dato que hay testing, estos valores son ahora los oficiales
    ganancia_normalizada  <- tbl[ , max(gan_suavizada, na.rm=TRUE) ]
    cant_corte  <- which.max( tbl[ , gan_suavizada ] ) 

    rm( tbl )
    gc()
  }



  #voy grabando las mejores column importance
  if( ganancia_normalizada >  GLOBAL_ganancia )
  {
    GLOBAL_ganancia  <<- ganancia_normalizada

    param_impo <-  copy( param_completo )
    param_impo$early_stopping_rounds  <- 0
    param_impo$num_iterations  <- modelocv$best_iter

    modelo  <- glm.train( data= dtrain,
                       param=  param_impo)                    

    tb_importancia  <- as.data.table( glm.importance( modelo ) )
    fwrite( tb_importancia,
            file= paste0( "impo_", GLOBAL_iteracion, ".txt" ),
            sep= "\t" )
    
    rm( tb_importancia )
  }


  #logueo final
  ds  <- list( "cols"= ncol(dtrain),  "rows"= nrow(dtrain) )
  xx  <- c( ds, copy(param_completo) )

  xx$early_stopping_rounds  <- NULL
  xx$num_iterations  <- modelocv$best_iter
  xx$estimulos   <- cant_corte
  xx$ganancia  <- ganancia_normalizada
  xx$iteracion_bayesiana  <- GLOBAL_iteracion

  exp_log( xx,  arch= "BO_log.txt" )

  return( ganancia_normalizada )
}

#------------------------------------------------------------------------------
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

#la particion de train siempre va
dtrain  <- glm.Dataset( data=    data.matrix( dataset[ fold_train==1, campos_buenos, with=FALSE] ),
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
  dvalidate  <- glm.Dataset( data=  data.matrix( dataset[ fold_validate==1, campos_buenos, with=FALSE] ),
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


#si ya existe el archivo log, traigo hasta donde procese
if( file.exists( "BO_log.txt" ) )
{
  tabla_log  <- fread( "BO_log.txt" )
  GLOBAL_iteracion  <- nrow( tabla_log )
  GLOBAL_ganancia   <- tabla_log[ , max(ganancia) ]
  rm(tabla_log)
} else  {
  GLOBAL_iteracion  <- 0
  GLOBAL_ganancia   <- -Inf
}


#Aqui comienza la configuracion de mlrMBO para el proceso de optimización

#debo hacer cross validation o  Train/Validate/Test
if( kcrossvalidation ) {
  funcion_optimizar  <- EstimarGanancia_glmCV
} else {
  #***funcion_optimizar  <- EstimarGanancia_lightgbm
}


configureMlr( show.learner.output= FALSE)

#configuro la busqueda bayesiana,  los hiperparametros que se van a optimizar
#por favor, no desesperarse por lo complejo
obj.fun  <- makeSingleObjectiveFunction(
              fn=       funcion_optimizar, #la funcion que voy a maximizar
              minimize= FALSE,   #estoy Maximizando la ganancia
              noisy=    TRUE,
              par.set=  PARAM$bo_glm,     #definido al comienzo del programa
              has.simple.signature = FALSE   #paso los parametros en una lista
             )

#archivo donde se graba y cada cuantos segundos
ctrl  <- makeMBOControl( save.on.disk.at.time= 600,  
                         save.file.path=       "bayesiana.RDATA" )

ctrl  <- setMBOControlTermination( ctrl, 
                                   iters= PARAM$bo_iteraciones )   #cantidad de iteraciones

ctrl  <- setMBOControlInfill(ctrl, crit= makeMBOInfillCritEI() )

#establezco la funcion que busca el maximo
surr.km  <- makeLearner("regr.km",
                        predict.type= "se",
                        covtype= "matern3_2",
                        control= list(trace= TRUE) )



#Aqui inicio la optimizacion bayesiana
if( !file.exists( "bayesiana.RDATA" ) ) {

  run  <- mbo(obj.fun, learner= surr.km, control= ctrl)

} else {
  #si ya existe el archivo RDATA, debo continuar desde el punto hasta donde llegue
  #  usado para cuando se corta la virtual machine
  run  <- mboContinue( "bayesiana.RDATA" )   #retomo en caso que ya exista
}

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
