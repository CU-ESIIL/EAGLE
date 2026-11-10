# Standard lidar metrics per cookie with lidR (called by 05_lidr_metrics.py; one batch of cookies per call).
#   Rscript scripts/baseline_eval/lidr_metrics.R <points.parquet> <out.parquet>
# Input: long table of points (site, X, Y, Z = height above ground, Zabs = elevation, Intensity, ReturnNumber,
# NumberOfReturns, Classification), already cropped to the analysis disc.
# Output: one row per site:
#   - lidR::stdmetrics (Roussel et al. 2020) on height above ground, defaults dz = 1, th = 2: columns as named by lidR
#     (zmax, zmean, zsd, zskew, zkurt, zentropy, pzabovezmean, pzabove2, zq5..zq95, zpcum1..9, itot, imax, imean,
#     isd, iskew, ikurt, ipground, ipcumzq10..90, p1th..p5th, pground, n, area)
#   - terrain_* from the ground points (class 2): lidR::rasterize_terrain(res = 2, tin()) then terra::terrain on the
#     DTM, averaged over the disc: elevation mean / range, slope (deg), northness = cos(aspect) * sin(slope),
#     eastness = sin(aspect) * sin(slope), TRI, TPI; and n_ground.
suppressMessages({
  library(lidR)
  library(arrow)
  library(data.table)
  library(terra)
})
args <- commandArgs(trailingOnly = TRUE)
pts <- as.data.table(read_parquet(args[1]))
sites <- unique(pts$site)

terrain_metrics <- function(d) {
  g <- d[Classification == 2L]
  out <- list(n_ground = nrow(g), terrain_elev_mean = NA_real_, terrain_elev_range = NA_real_,
              terrain_slope = NA_real_, terrain_northness = NA_real_, terrain_eastness = NA_real_,
              terrain_tri = NA_real_, terrain_tpi = NA_real_)
  if (nrow(g) < 50) return(out)
  las <- suppressMessages(LAS(data.table(X = d$X, Y = d$Y, Z = d$Zabs, Classification = d$Classification)))
  dtm <- tryCatch(suppressMessages(suppressWarnings(rasterize_terrain(las, res = 2, algorithm = tin()))),
                  error = function(e) NULL)
  if (is.null(dtm)) return(out)
  tr <- terra::terrain(dtm, v = c("slope", "aspect", "TRI", "TPI"), unit = "radians")
  v <- terra::values(c(dtm, tr), mat = TRUE)
  v <- v[stats::complete.cases(v), , drop = FALSE]
  if (nrow(v) == 0) return(out)
  out$terrain_elev_mean <- mean(v[, 1]); out$terrain_elev_range <- diff(range(v[, 1]))
  out$terrain_slope <- mean(v[, "slope"]) * 180 / pi
  out$terrain_northness <- mean(cos(v[, "aspect"]) * sin(v[, "slope"]))
  out$terrain_eastness <- mean(sin(v[, "aspect"]) * sin(v[, "slope"]))
  out$terrain_tri <- mean(v[, "TRI"]); out$terrain_tpi <- mean(v[, "TPI"])
  out
}

rows <- vector("list", length(sites))
for (k in seq_along(sites)) {
  d <- pts[site == sites[k]]
  std <- tryCatch(
    stdmetrics(d$X, d$Y, d$Z, d$Intensity, d$ReturnNumber, d$Classification, dz = 1, th = 2),
    error = function(e) list(stdmetrics_error = conditionMessage(e)))
  rows[[k]] <- c(list(als_site_id = sites[k]), std, terrain_metrics(d))
}
write_parquet(rbindlist(rows, fill = TRUE), args[2])
