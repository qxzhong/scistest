#!/usr/bin/env Rscript
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) stop("Usage: run.R METHOD INPUT X_COLUMNS SEED OUTPUT")
method <- args[[1]]
if (method == "vim") {
  suppressPackageStartupMessages(library(survML))
  suppressPackageStartupMessages(library(SuperLearner))
  suppressPackageStartupMessages(library(gam))
  suppressPackageStartupMessages(library(nnls))
} else {
  suppressPackageStartupMessages(library(comets))
}
run_vim <- function(dat, x_cols) {
  X <- dat[, paste0("W", 1:10), drop = FALSE]
  small_feature_vector <- setdiff(seq_len(ncol(X)), match(x_cols, names(X)))
  output <- vim(
    type = "AUC", time = dat$T, event = dat$Delta, X = X,
    landmark_times = stats::quantile(dat$T[dat$Delta == 1], probs = c(0.9)),
    large_feature_vector = seq_len(ncol(X)), small_feature_vector = small_feature_vector,
    conditional_surv_generator_control = list(SL.library = c("SL.glm", "SL.glm")),
    large_oracle_generator_control = list(SL.library = c("SL.glm", "SL.glm")),
    small_oracle_generator_control = list(SL.library = c("SL.glm", "SL.glm")),
    cf_fold_num = 2, sample_split = TRUE, scale_est = TRUE)
  as.numeric(output$result$p)
}

run_complete_case <- function(dat, x_cols, method) {
  complete <- dat[dat$Delta == 1, , drop = FALSE]
  predictor_cols <- setdiff(names(dat), c("T", "Delta"))
  z_cols <- setdiff(predictor_cols, x_cols)
  x <- as.matrix(complete[, x_cols, drop = FALSE])
  z <- as.matrix(complete[, z_cols, drop = FALSE])
  y <- complete$T
  fit <- if (method == "GCM") {
    comets::gcm(Y = y, X = x, Z = z, reg_YonZ = "rf", reg_XonZ = "rf")
  } else {
    comets::pcm(Y = y, X = x, Z = z, reg_YonZ = "rf", reg_XonZ = "rf")
  }
  as.numeric(fit$p.value)
}

dat <- read.csv(args[[2]], check.names = FALSE)
x_cols <- strsplit(args[[3]], ";", fixed = TRUE)[[1]]
set.seed(as.integer(as.numeric(args[[4]]) %% .Machine$integer.max))
p <- if (method == "vim") run_vim(dat, x_cols) else run_complete_case(dat, x_cols, method)
if (length(p) != 1 || !is.finite(p) || p < 0 || p > 1) stop("Invalid p-value")
writeLines(sprintf("%.17g", p), args[[5]])
