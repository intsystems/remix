# ReMix

## Основные классы

`Tensor` в сигнатурах &mdash; тензор PyTorch. Все интерфейсы ниже планируются к реализации.

| Класс | Основные методы |
|---|---|
| `GaussianMixture(logits: Tensor, loc: Tensor, scale: Tensor)` | Хранит параметры смеси. `sample(sample_shape: torch.Size) -> Tensor` генерирует сэмплы; `log_prob(z: Tensor) -> Tensor` считает логарифм плотности; `responsibilities(z: Tensor) -> Tensor` возвращает вероятности принадлежности компонентам. |
| `DiagonalGaussianMixture(n_components: int, event_dim: int)` | Модуль с обучаемыми параметрами. `distribution() -> GaussianMixture` создаёт распределение для текущего шага. Положительные значения получаем через `softplus`. |
| `ELBOObjective(log_joint: Callable, stl: bool = False)` | `evaluate(q: GaussianMixture, z: Tensor) -> Tensor` вычисляет ELBO: `log_joint(z) - log_q(z)`. При STL отключает прямой градиент по параметрам плотности `q`, сохраняя градиент через `z`. |
| `GradientEstimator(n_samples: int = 1)` | Общий интерфейс: `estimate(q: GaussianMixture, objective: ELBOObjective) -> EstimateResult`. Результат содержит `value: Tensor` для логирования и скаляр `surrogate_loss: Tensor` для обучения. |

Реализации GradientEstimator:

- **`ScoreFunction`** &mdash; [статья](https://arxiv.org/abs/1703.09194).
- **`Stratified`** &mdash; [статья](https://www.jmlr.org/papers/v27/25-2560.html).
- **`Implicit`** &mdash; [статья](https://arxiv.org/abs/1805.08498).
- **`PostStratified`** &mdash; [статья](https://www.jmlr.org/papers/v27/25-2560.html).



## Стек

**Стек:** Python, PyTorch (`torch.distributions`, autograd, Adam, Pyro), pytest, Matplotlib; torchvision для датасетов с изображениями.
