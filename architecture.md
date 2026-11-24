# ReMix

## Основные классы

`Tensor` в сигнатурах &mdash; тензор PyTorch. Публичный API использует
`Mixture(logits, components)`: компоненты должны быть несмешанными
(`batch_shape` пустой) и иметь одинаковую форму события.

| Класс | Основные методы |
|---|---|
| `Mixture(logits: Tensor, components: Sequence[Distribution])` | `sample`, `log_prob`, `responsibilities` и `detach` для конечной смеси. Параметры компонентов задаются объектами PyTorch Distribution. |
| `ELBOObjective(log_joint: Callable, stl: bool = False)` | `evaluate(q: Mixture, z: Tensor) -> Tensor` вычисляет `log_joint(z) - log_q(z)`. STL отключает прямой градиент через параметры плотности `q`, сохраняя градиент через `z`. |
| `GradientEstimator(n_samples: int = 1)` | Общий интерфейс: `estimate(q: Mixture, objective: ELBOObjective) -> EstimateResult`. Результат содержит `value` для логирования и скалярный `surrogate_loss` для обучения. |

Реализации `GradientEstimator`, доступные из публичного API:

- **`ScoreFunction`** &mdash; [статья](https://arxiv.org/abs/1703.09194).
- **`Stratified`** &mdash; [статья](https://www.jmlr.org/papers/v27/25-2560.html).
- **`Implicit`** &mdash; для скалярных непрерывных смесей с дифференцируемыми CDF компонент; [статья](https://arxiv.org/abs/1805.08498).
- **`PostStratified`** &mdash; для скалярных непрерывных смесей с дифференцируемыми CDF и обратными CDF компонент; [статья](https://www.jmlr.org/papers/v27/25-2560.html).

`Implicit` и `PostStratified` пока ограничены скалярными событиями.
`Stratified` требует `rsample()` у каждой компоненты; `Implicit` требует
дифференцируемую `cdf()`, а `PostStratified` &mdash; дифференцируемые
`cdf()` и `icdf()`.



## Стек

**Стек:** Python, PyTorch (`torch.distributions`, autograd, Adam, Pyro), pytest, Matplotlib; torchvision для датасетов с изображениями.
