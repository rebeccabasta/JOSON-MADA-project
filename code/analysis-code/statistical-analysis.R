###############################
# Analysis script
#
# This script loads the cleaned peanut variety dataset,
# performs statistical tests, unsupervised clustering,
# and supervised machine learning model comparison.
###############################

library(tidyverse)
library(tidymodels)
library(broom)
library(here)
library(factoextra)
library(vip)

set.seed(123)

# Load cleaned data
data_location <- here("data", "processed-data", "processeddata.rds")
clean_data <- readRDS(data_location)

# Check data structure
glimpse(clean_data)
summary(clean_data)

######################################
# Classical statistical analysis
######################################

# Effect of variety on yield
model_variety <- aov(peanut_yield ~ variety, data = clean_data)
model_1 <- broom::tidy(model_variety)

saveRDS(
  model_1,
  file = here("results", "tables", "resulttable_variety_aov.rds")
)

# Effect of location on yield
model_location <- aov(peanut_yield ~ location, data = clean_data)
model_2 <- broom::tidy(model_location)

saveRDS(
  model_2,
  file = here("results", "tables", "resulttable_location_aov.rds")
)

# Effect of irrigation/watering on yield
model_watered <- t.test(peanut_yield ~ watered, data = clean_data)
model_3 <- broom::tidy(model_watered)

saveRDS(
  model_3,
  file = here("results", "tables", "resulttable_watered_ttest.rds")
)

######################################
# Unsupervised learning: variety clustering
######################################

# Summarize yield traits by variety
variety_summary <- clean_data %>%
  group_by(variety) %>%
  summarise(
    avg_yield = mean(peanut_yield, na.rm = TRUE),
    sd_yield = sd(peanut_yield, na.rm = TRUE),
    irrigated_yield = mean(peanut_yield[watered == TRUE], na.rm = TRUE),
    non_irrigated_yield = mean(peanut_yield[watered == FALSE], na.rm = TRUE),
    yield_diff = irrigated_yield - non_irrigated_yield,
    n_obs = n(),
    .groups = "drop"
  ) %>%
  filter(n_obs >= 50)

# Scale numeric variables before clustering
variety_scaled <- variety_summary %>%
  select(avg_yield, sd_yield, irrigated_yield, non_irrigated_yield, yield_diff) %>%
  scale()

# Elbow plot for choosing k
elbow_plot <- fviz_nbclust(variety_scaled, kmeans, method = "wss") +
  labs(
    title = "Elbow Method for Choosing Number of Clusters",
    x = "Number of clusters",
    y = "Within-cluster sum of squares"
  )

ggsave(
  filename = here("results", "figures", "elbow_plot.png"),
  plot = elbow_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# K-means clustering
set.seed(123)

kmeans_fit <- kmeans(variety_scaled, centers = 4)

variety_summary <- variety_summary %>%
  mutate(cluster = as.factor(kmeans_fit$cluster))

# PCA visualization of clusters
pca_plot <- fviz_cluster(
  kmeans_fit,
  data = variety_scaled,
  geom = "point",
  ellipse.type = "convex",
  ggtheme = theme_minimal()
) +
  labs(
    title = "PCA Visualization of Peanut Variety Clusters"
  )

ggsave(
  filename = here("results", "figures", "pca_cluster_plot.png"),
  plot = pca_plot,
  width = 7,
  height = 5,
  dpi = 300
)

# Summarize cluster characteristics
cluster_summary <- variety_summary %>%
  group_by(cluster) %>%
  summarise(
    avg_yield = mean(avg_yield),
    sd_yield = mean(sd_yield),
    yield_diff = mean(yield_diff),
    n_varieties = n(),
    .groups = "drop"
  )

saveRDS(
  cluster_summary,
  file = here("results", "tables", "cluster_summary.rds")
)

print(cluster_summary)

######################################
# Supervised machine learning: predicting peanut yield
######################################

# Prepare data for modeling
ml_data <- clean_data %>%
  mutate(
    year = as.factor(year),
    location = as.factor(location),
    watered = as.factor(watered),
    variety = as.factor(variety)
  )

# Train/test split
set.seed(123)

data_split <- initial_split(ml_data, prop = 0.80, strata = peanut_yield)

train_data <- training(data_split)
test_data  <- testing(data_split)

# Cross-validation folds
set.seed(123)

cv_folds <- vfold_cv(train_data, v = 5, repeats = 5, strata = peanut_yield)

# Recipe
yield_recipe <- recipe(peanut_yield ~ year + location + watered + variety,
                       data = train_data) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_zv(all_predictors()) %>%
  step_normalize(all_numeric_predictors())

#Model 1: Linear Regression Baseline
lm_spec <- linear_reg() %>%
  set_engine("lm")

lm_wf <- workflow() %>%
  add_recipe(yield_recipe) %>%
  add_model(lm_spec)

#Model 2: LASSO regression
lasso_spec <- linear_reg(
  penalty = tune(),
  mixture = 1
) %>%
  set_engine("glmnet")

lasso_wf <- workflow() %>%
  add_recipe(yield_recipe) %>%
  add_model(lasso_spec)

lasso_grid <- grid_regular(
  penalty(range = c(-5, 1)),
  levels = 20
)

#Model 3: Random Forest
rf_spec <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 500
) %>%
  set_engine("ranger", importance = "impurity") %>%
  set_mode("regression")

rf_wf <- workflow() %>%
  add_recipe(yield_recipe) %>%
  add_model(rf_spec)

rf_grid <- grid_regular(
  mtry(range = c(1, 20)),
  min_n(range = c(2, 20)),
  levels = 5
)

#Tune and compare models
# Linear model CV
lm_res <- fit_resamples(
  lm_wf,
  resamples = cv_folds,
  metrics = metric_set(rmse, rsq, mae)
)

# LASSO tuning
lasso_res <- tune_grid(
  lasso_wf,
  resamples = cv_folds,
  grid = lasso_grid,
  metrics = metric_set(rmse, rsq, mae)
)

# Random forest tuning
rf_res <- tune_grid(
  rf_wf,
  resamples = cv_folds,
  grid = rf_grid,
  metrics = metric_set(rmse, rsq, mae)
)

#model comparison results
lm_metrics <- collect_metrics(lm_res) %>%
  mutate(model = "Linear regression")

lasso_metrics <- collect_metrics(lasso_res) %>%
  mutate(model = "LASSO")

rf_metrics <- collect_metrics(rf_res) %>%
  mutate(model = "Random forest")

model_comparison <- bind_rows(
  lm_metrics,
  lasso_metrics,
  rf_metrics
)

saveRDS(
  model_comparison,
  file = here("results", "tables", "model_comparison_cv.rds")
)

print(model_comparison)

#select best model
best_lasso <- select_best(lasso_res, metric = "rmse")
best_rf <- select_best(rf_res, metric = "rmse")

best_lasso
best_rf

#Final model evaluation on test data
# Finalize workflows
final_lasso_wf <- finalize_workflow(lasso_wf, best_lasso)
final_rf_wf <- finalize_workflow(rf_wf, best_rf)

# Fit final models to training data and evaluate on test data
final_lasso_fit <- last_fit(
  final_lasso_wf,
  split = data_split,
  metrics = metric_set(rmse, rsq, mae)
)

final_rf_fit <- last_fit(
  final_rf_wf,
  split = data_split,
  metrics = metric_set(rmse, rsq, mae)
)

# Collect final test metrics
lasso_test_metrics <- collect_metrics(final_lasso_fit) %>%
  mutate(model = "LASSO")

rf_test_metrics <- collect_metrics(final_rf_fit) %>%
  mutate(model = "Random forest")

test_model_comparison <- bind_rows(
  lasso_test_metrics,
  rf_test_metrics
)

saveRDS(
  test_model_comparison,
  file = here("results", "tables", "test_model_comparison.rds")
)

print(test_model_comparison)

#variable importance for RF
final_rf_model <- extract_workflow(final_rf_fit) %>%
  extract_fit_parsnip()

rf_importance_plot <- vip(final_rf_model, num_features = 15) +
  labs(
    title = "Random Forest Variable Importance",
    x = "Importance",
    y = "Predictor"
  )

ggsave(
  filename = here("results", "figures", "rf_variable_importance.png"),
  plot = rf_importance_plot,
  width = 8,
  height = 6,
  dpi = 300
)

