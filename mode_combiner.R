# ===========================
# Bio3D mode combiner (R)
# ===========================
# Requirements: bio3d
# install.packages("bio3d")

# ---------- Small utilities ----------
as_xyz_mat <- function(v) {
  stopifnot(length(v) %% 3 == 0)
  matrix(v, ncol = 3, byrow = TRUE)
}
as_xyz_vec <- function(M) as.vector(t(M))
norm_vec   <- function(v) v / sqrt(sum(v^2))

# Build CA selection + key ("CHAIN:RESNO" or "RESNO")
make_ca_key <- function(atom, by_chain = TRUE) {
  sel <- atom$elety == "CA"
  key <- if (by_chain) paste0(atom$chain[sel], ":", atom$resno[sel]) else as.character(atom$resno[sel])
  list(sel = sel, key = key)
}

# ---------- Interactive pickers ----------
# Choose a single PDB from a directory (numbered menu)
choose_pdb_file <- function(directory = ".", pattern = "\\.pdb$") {
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("No PDB files found in ", normalizePath(directory))
  cat("\nAvailable PDB files in", normalizePath(directory), ":\n")
  for (i in seq_along(files)) cat(sprintf("[%d] %s\n", i, basename(files[i])))
  repeat {
    choice <- as.integer(readline("Select reference PDB by number: "))
    if (!is.na(choice) && choice >= 1 && choice <= length(files)) break
    cat("Invalid choice. Enter 1..", length(files), "\n")
  }
  sel <- normalizePath(files[choice])
  cat("Selected:", basename(sel), "\n\n")
  sel
}

# Choose multiple mode PDBs (returns named vector; names are extracted mode numbers)
choose_mode_files <- function(directory = ".", pattern = "^mode[0-9]+.*\\.pdb$") {
  files <- list.files(directory, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) stop("No mode PDBs matching ", pattern, " in ", normalizePath(directory))
  base <- basename(files)
  mode_nums <- sub("^mode([0-9]+).*\\.pdb$", "\\1", base)
  cat("\nAvailable mode PDBs:\n")
  for (i in seq_along(files)) cat(sprintf("[%d] %s  (mode %s)\n", i, base[i], mode_nums[i]))
  cat("Enter numbers to select (comma/space separated), e.g. 1,3,5: ")
  txt <- readline()
  picks <- as.integer(unlist(strsplit(gsub("[, ]+", " ", txt), " ")))
  picks <- picks[!is.na(picks) & picks >= 1 & picks <= length(files)]
  if (length(picks) == 0) stop("No valid selections.")
  sel <- normalizePath(files[picks])
  names(sel) <- mode_nums[picks]
  cat("Selected modes:", paste(names(sel), collapse = ", "), "\n\n")
  sel
}

# ---------- PDB multi-MODEL handling ----------
split_pdb_models <- function(pdb_path) {
  lines <- readLines(pdb_path, warn = FALSE)
  starts <- grep("^MODEL",  lines)
  ends   <- grep("^ENDMDL", lines)
  if (length(starts) == 0 || length(ends) == 0) {
    return(list(lines))  # treat as single-model PDB
  }
  L <- min(length(starts), length(ends))
  models <- vector("list", L)
  for (i in seq_len(L)) models[[i]] <- lines[starts[i]:ends[i]]
  models
}

# Reduce a multi-MODEL PDB to a single unit direction vector in 3N space
extract_mode_vector_from_multimodel <- function(ref_obj,
                                                mode_pdb_path,
                                                by_chain = TRUE,
                                                strategy = c("max", "svd"),
                                                normalize = TRUE) {
  strategy <- match.arg(strategy)

  # Ensure ref has xyz vector
  ref_obj$xyz <- if (is.null(ref_obj$xyz)) as_xyz_vec(as.matrix(ref_obj$atom[, c("x","y","z")])) else ref_obj$xyz

  # Reference CA mapping
  refk <- make_ca_key(ref_obj$atom, by_chain = by_chain)
  idx_ca <- which(refk$sel); key_ref <- refk$key
  if (length(idx_ca) < 3) stop("Reference has too few Cα atoms.")

  mdl_lines <- split_pdb_models(mode_pdb_path)
  if (length(mdl_lines) == 0) stop("No models found in: ", mode_pdb_path)

  # For each MODEL: compute CA displacement vs ref on matched residues; expand to 3N
  library(bio3d)
  disp_cols <- list()
  for (i in seq_along(mdl_lines)) {
    tf <- tempfile(fileext = ".pdb")
    writeLines(mdl_lines[[i]], tf)
    mov <- read.pdb(tf)
    mov$xyz <- if (is.null(mov$xyz)) as_xyz_vec(as.matrix(mov$atom[, c("x","y","z")])) else mov$xyz

    movk <- make_ca_key(mov$atom, by_chain = by_chain)
    idx_ca_m <- which(movk$sel); key_mov <- movk$key

    common <- intersect(key_ref, key_mov)
    if (length(common) < 3) next

    i_ref <- idx_ca[   match(common, key_ref) ]
    i_mov <- idx_ca_m[ match(common, key_mov) ]

    ref_mat <- as_xyz_mat(ref_obj$xyz)[i_ref, , drop = FALSE]
    mov_mat <- as_xyz_mat(mov$xyz)[i_mov, , drop = FALSE]
    disp_CA <- as_xyz_vec(mov_mat - ref_mat)  # 3 * n_common

    disp_full <- numeric(length(ref_obj$xyz))
    pos <- as.vector(rbind(3*i_ref - 2, 3*i_ref - 1, 3*i_ref))
    disp_full[pos] <- disp_CA
    disp_cols[[length(disp_cols) + 1]] <- disp_full
  }

  if (length(disp_cols) == 0) stop("No usable models after matching CA residues: ", mode_pdb_path)

  B <- do.call(cbind, disp_cols)  # (3N) x L

  if (strategy == "max") {
    j <- which.max(colSums(B^2))
    v <- B[, j]
  } else { # "svd"
    sv <- svd(B, nu = 1, nv = 0)
    v <- sv$u[, 1]
  }

  if (normalize) v <- norm_vec(v)
  v
}

# ---------- Main combiner ----------
# If ref_pdb_file is NULL/missing, a menu will prompt you to pick a reference PDB.
# If mode_files is missing or length 0, a menu will prompt you to pick mode PDBs.
# 'coeffs' must be a named numeric vector whose names match names(mode_files) (mode numbers).
combine_modes_from_multimodels <- function(ref_pdb_file = NULL,
                                           mode_files = NULL,   # named vec: c("7"="mode7.pdb", ...)
                                           coeffs     = NULL,   # named weights, same names as mode_files
                                           by_chain   = TRUE,
                                           normalize  = TRUE,   # combine normalized per-mode directions
                                           per_mode_strategy = c("max","svd"),
                                           amp        = 1.0,
                                           out_pdb    = "combined_modes.pdb",
                                           picker_dir = ".") {
  per_mode_strategy <- match.arg(per_mode_strategy)

  # --- reference selection ---
  if (is.null(ref_pdb_file) || !file.exists(ref_pdb_file)) {
    message("\nNo reference PDB provided — please select one:")
    ref_pdb_file <- choose_pdb_file(directory = picker_dir, pattern = "\\.pdb$")
  }

  # --- mode files selection (optional interactive) ---
  if (is.null(mode_files) || length(mode_files) == 0) {
    message("No mode files provided — please select them:")
    mode_files <- choose_mode_files(directory = picker_dir, pattern = "^mode[0-9]+.*\\.pdb$")
  }

  # checks
  exists_mask <- file.exists(mode_files)
  if (!all(exists_mask)) {
    missing <- paste0(names(mode_files)[!exists_mask], " -> ", mode_files[!exists_mask])
    stop("These mode PDB files are missing:\n", paste(missing, collapse = "\n"))
  }

  if (is.null(coeffs)) {
    # default: weight 1.0 for each selected mode
    coeffs <- setNames(rep(1.0, length(mode_files)), names(mode_files))
  }
  stopifnot(!is.null(names(mode_files)), !is.null(names(coeffs)))
  stopifnot(setequal(names(mode_files), names(coeffs)))

  # --- load reference ---
  library(bio3d)
  ref <- read.pdb(ref_pdb_file)
  ref$xyz <- if (is.null(ref$xyz)) as_xyz_vec(as.matrix(ref$atom[, c("x","y","z")])) else ref$xyz

  # --- extract per-mode vectors from (possibly) multi-MODEL PDBs ---
  ord <- names(mode_files)
  mode_vecs <- vector("list", length(ord))
  for (i in seq_along(ord)) {
    mode_vecs[[i]] <- extract_mode_vector_from_multimodel(
      ref_obj     = ref,
      mode_pdb_path = mode_files[[i]],
      by_chain    = by_chain,
      strategy    = per_mode_strategy,
      normalize   = TRUE
    )
  }

  # --- combine linearly ---
  B <- do.call(cbind, mode_vecs)         # (3N) x K
  w <- as.numeric(coeffs[ord])           # K
  disp <- as.vector(B %*% w)             # combined 3N displacement
  if (!normalize) {
    # If you truly want raw per-mode lengths, re-run extract_mode_vector_from_multimodel
    # with normalize=FALSE and set normalize=FALSE here. Default is safer (direction-only).
  }

  new_xyz <- ref$xyz + amp * disp
  per_atom_mag <- sqrt(rowSums(as_xyz_mat(disp)^2))

  out <- ref
  out$xyz <- new_xyz
  out$atom$b <- per_atom_mag
  bio3d::write.pdb(out, file = out_pdb)

  message("✅ Wrote combined structure to: ", normalizePath(out_pdb))
  invisible(list(
    disp      = disp,
    modes     = ord,
    weights   = w,
    strategy  = per_mode_strategy,
    by_chain  = by_chain,
    out_file  = normalizePath(out_pdb)
  ))
}

# ---------- Example usage ----------
# 1) Interactive reference + interactive mode selection, unit weights:
# res <- combine_modes_from_multimodels(
#   ref_pdb_file = NULL, # nolint: commented_code_linter.
#   mode_files   = NULL, # nolint
#   coeffs       = NULL,           # defaults to weight 1.0 each
#   by_chain     = TRUE,           # set FALSE if chain IDs differ
#   per_mode_strategy = "max",     # or "svd"
#   amp          = 1.2,
#   out_pdb      = "combined_from_menu.pdb",
#   picker_dir   = "."             # where to list files from
# )
#
# 2) Programmatic call with explicit files/weights:
# ref <- "reference.pdb"
# mode_files <- c("7"="mode_7.pdb", "8"="mode_8.pdb") # nolint
# coeffs     <- c("7"= 1.0, "8"= -0.5)
# res <- combine_modes_from_multimodels(
#   ref_pdb_file = ref,
#   mode_files   = mode_files, # nolint: commented_code_linter.
#   coeffs       = coeffs, # nolint
#   by_chain     = TRUE,
#   per_mode_strategy = "svd",
#   amp          = 1.0,
#   out_pdb      = "combined_7_8.pdb"
# )

