options(timeout = 1800, repos = c(CRAN = "https://mirrors.tuna.tsinghua.edu.cn/CRAN"))
library_path <- Sys.getenv("R_LIBS_USER")
if (!nzchar(library_path)) library_path <- path.expand("~/R/scistest-library")
dir.create(library_path, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(library_path, .libPaths()))

if (!requireNamespace("remotes", quietly = TRUE)) {
  install.packages("remotes", lib = library_path, Ncpus = 16)
}
if (!requireNamespace("jsonlite", quietly = TRUE)) {
  install.packages("jsonlite", lib = library_path, Ncpus = 16)
}

required <- c(
  SuperLearner = "2.0-41",
  nnls = "1.6",
  gam = "1.22-7",
  foreach = "1.5.2",
  randomForest = "4.7-1.2",
  earth = "5.3.6",
  ranger = "0.18.0",
  survML = "1.2.0",
  comets = "0.2-2"
)

for (package in names(required)) {
  installed <- requireNamespace(package, quietly = TRUE)
  observed <- if (installed) utils::packageDescription(package, fields = "Version") else NA_character_
  correct <- installed && identical(unname(observed), required[[package]])
  if (!correct) {
    remotes::install_version(package, version = required[[package]], lib = library_path,
                             dependencies = c("Depends", "Imports", "LinkingTo"),
                             upgrade = "never", quiet = FALSE)
  }
}

for (package in names(required)) {
  stopifnot(requireNamespace(package, quietly = TRUE))
  observed <- unname(utils::packageDescription(package, fields = "Version"))
  if (!identical(observed, required[[package]])) {
    stop(sprintf("%s version %s; expected %s", package, observed, required[[package]]))
  }
}
writeLines(capture.output(sessionInfo()), file.path(tempdir(), "scistest_r_session_info.txt"))
