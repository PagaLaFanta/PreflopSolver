//! # Pool de hilos: envoltorio fino sobre `rayon`
//!
//! Hasta ahora este módulo era un pool a mano (mutex + condvar + un contador "idle" como
//! heurística de reparto, ver historial de git para esa versión). Funcionaba, pero con un
//! árbol **estrecho** (pocas acciones por nodo — p. ej. un preflop sin ICM con pocos
//! tamaños de 3-bet) el reparto se venía abajo: en cuanto `join`/`join_all` veía el
//! contador "idle" a cero en el instante exacto de repartir, ejecutaba **todo** en línea en
//! vez de encolarlo — todo o nada, sin forma de que un hilo que acabó pronto robara trabajo
//! de una cola más profunda. Medido en un 3-max sin ICM, Chip EV, con 3-bets all-in y no
//! all-in: el Best Response (el paso caro de NashConv) usaba ~19 % de 16 hilos lógicos y
//! tardaba ~90 s por el mismo motivo.
//!
//! `rayon` resuelve esto con colas de robo de trabajo (work-stealing) de verdad: cada hilo
//! tiene su propia cola y roba de las de los demás cuando se queda sin trabajo propio, sin
//! el "todo o nada" de la heurística anterior. [`Pool`] se queda solo como un marcador
//! (`parallel: bool`) para que `join`/`join_all` con `n ≤ 1` sigan corriendo en línea sin
//! tocar rayon; con `n > 1`, delegan directamente en `rayon::join`/`rayon::scope`.
//!
//! [`with_pool!`] arranca un `rayon::ThreadPool` de `n` hilos (una vez por llamada — típicamente
//! una vez por todo un `Solver::run(iters)`, no por nodo) y ejecuta `$body` instalado en él.

use std::marker::PhantomData;

/// Mango al pool activo. Solo recuerda si hay que paralelizar; el reparto real (si
/// `parallel`) lo hace el `rayon::ThreadPool` instalado por [`with_pool!`] en la pila de
/// llamadas actual (rayon lo encuentra solo, no hace falta pasarlo aquí).
#[derive(Clone, Copy, Debug)]
pub struct Pool<'a> {
    parallel: bool,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Pool<'a> {
    pub fn new(parallel: bool) -> Self {
        Pool { parallel, _marker: PhantomData }
    }

    /// Ejecuta `a` y `b` en paralelo si hay pool; si no, `a()` y luego `b()` en línea.
    pub fn join<RA: Send, RB: Send>(&self, a: impl FnOnce() -> RA + Send, b: impl FnOnce() -> RB + Send) -> (RA, RB) {
        if self.parallel {
            rayon::join(a, b)
        } else {
            (a(), b())
        }
    }

    /// Como [`Self::join`] con `n` tareas en vez de 2. Resultados en el mismo orden que
    /// `tasks`. Con pool, reparte con `rayon::scope` (espera a que todas acaben, incluido
    /// si alguna entra en pánico — el pánico se relanza aquí, como antes).
    pub fn join_all<R: Send>(&self, tasks: Vec<Box<dyn FnOnce() -> R + Send + 'a>>) -> Vec<R> {
        if !self.parallel || tasks.len() <= 1 {
            return tasks.into_iter().map(|t| t()).collect();
        }
        let mut slots: Vec<Option<R>> = (0..tasks.len()).map(|_| None).collect();
        {
            let slots = &mut slots;
            rayon::scope(|s| {
                for (slot, task) in slots.iter_mut().zip(tasks) {
                    s.spawn(move |_| *slot = Some(task()));
                }
            });
        }
        slots.into_iter().map(|r| r.expect("rayon::scope no esperó a una tarea de join_all")).collect()
    }
}

/// Arranca un `rayon::ThreadPool` de `n` hilos y ejecuta `$body` instalado en él, con
/// `$pool: Pool` a su disposición. Con `n ≤ 1`, corre `$body` en línea sin construir pool
/// (mismo atajo que antes: no vale la pena el coste de arrancar hilos para un recorrido
/// que de todas formas va a ejecutarse secuencial).
#[macro_export]
macro_rules! with_pool {
    ($n:expr, |$pool:ident| $body:expr) => {{
        let n = $n;
        if n <= 1 {
            let $pool = $crate::cfr::pool::Pool::new(false);
            $body
        } else {
            let rp = rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("construir el pool de rayon");
            rp.install(|| {
                let $pool = $crate::cfr::pool::Pool::new(true);
                $body
            })
        }
    }};
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn join_runs_both_and_returns_in_order() {
        crate::with_pool!(4, |pool| {
            let (a, b) = pool.join(|| 1 + 1, || 2 + 2);
            assert_eq!((a, b), (2, 4));
        });
    }

    #[test]
    fn join_all_preserves_order_under_load() {
        crate::with_pool!(4, |pool| {
            let tasks: Vec<Box<dyn FnOnce() -> usize + Send>> = (0..50usize).map(|i| Box::new(move || i * i) as Box<dyn FnOnce() -> usize + Send>).collect();
            let out = pool.join_all(tasks);
            assert_eq!(out, (0..50usize).map(|i| i * i).collect::<Vec<_>>());
        });
    }

    #[test]
    fn join_all_with_borrowed_data_not_static() {
        crate::with_pool!(4, |pool| {
            let local = vec![1, 2, 3, 4, 5];
            let local_ref = &local;
            let tasks: Vec<Box<dyn FnOnce() -> i32 + Send + '_>> = (0..local.len()).map(|i| Box::new(move || local_ref[i] * 10) as Box<dyn FnOnce() -> i32 + Send + '_>).collect();
            let out = pool.join_all(tasks);
            assert_eq!(out, vec![10, 20, 30, 40, 50]);
        });
    }

    #[test]
    fn nested_join_all_does_not_deadlock() {
        // Cada tarea de nivel 1 lanza su propio `join_all` de nivel 2 reutilizando el mismo
        // `pool` (recursión real, como en la recursión del árbol preflop).
        crate::with_pool!(3, |pool| {
            let tasks: Vec<Box<dyn FnOnce() -> usize + Send>> = (0..20usize)
                .map(|i| {
                    Box::new(move || {
                        let inner: Vec<Box<dyn FnOnce() -> usize + Send>> = (0..20usize).map(|j| Box::new(move || i * 100 + j) as Box<dyn FnOnce() -> usize + Send>).collect();
                        pool.join_all(inner).into_iter().sum()
                    }) as Box<dyn FnOnce() -> usize + Send>
                })
                .collect();
            let out = pool.join_all(tasks);
            let expected: Vec<usize> = (0..20usize).map(|i| (0..20usize).map(|j| i * 100 + j).sum()).collect();
            assert_eq!(out, expected);
        });
    }

    #[test]
    fn panic_in_a_task_propagates_without_hanging_siblings() {
        let result = std::panic::catch_unwind(|| {
            crate::with_pool!(4, |pool| {
                let tasks: Vec<Box<dyn FnOnce() -> usize + Send>> = vec![
                    Box::new(|| 1),
                    Box::new(|| panic!("boom")),
                    Box::new(|| 3),
                ];
                pool.join_all(tasks)
            })
        });
        assert!(result.is_err(), "el panic de una tarea debe propagarse, no perderse");
    }

    #[test]
    fn panic_in_the_inline_half_still_waits_for_the_queued_half() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag2 = Arc::clone(&flag);
        let result = std::panic::catch_unwind(|| {
            crate::with_pool!(4, |pool| {
                pool.join(
                    || panic!("boom en la mitad en línea"),
                    move || {
                        std::thread::sleep(Duration::from_millis(20));
                        flag2.store(true, Relaxed);
                    },
                )
            })
        });
        assert!(result.is_err());
        assert!(flag.load(Relaxed), "la tarea encolada debió terminar de ejecutarse antes de propagar el pánico");
    }

    #[test]
    fn with_pool_zero_or_one_threads_runs_inline() {
        for n in [0, 1] {
            crate::with_pool!(n, |pool| {
                let (a, b) = pool.join(|| 10, || 20);
                assert_eq!((a, b), (10, 20));
            });
        }
    }

    #[test]
    fn heavy_fan_out_uses_all_workers() {
        // No comprueba el paralelismo real (eso es cosa del benchmark manual), solo que
        // con trabajo real y muchas tareas, la cuenta de "ocupados simultáneos" máxima
        // observada se acerca al número de trabajadores, no se queda en 1.
        let concurrent = Arc::new(AtomicU32::new(0));
        let max_seen = Arc::new(AtomicU32::new(0));
        crate::with_pool!(6, |pool| {
            let tasks: Vec<Box<dyn FnOnce() + Send>> = (0..60)
                .map(|_| {
                    let concurrent = Arc::clone(&concurrent);
                    let max_seen = Arc::clone(&max_seen);
                    Box::new(move || {
                        let cur = concurrent.fetch_add(1, Relaxed) + 1;
                        max_seen.fetch_max(cur, Relaxed);
                        std::thread::sleep(Duration::from_millis(5));
                        concurrent.fetch_sub(1, Relaxed);
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            pool.join_all(tasks);
        });
        assert!(max_seen.load(Relaxed) >= 4, "esperaba varios trabajadores activos a la vez, vi {}", max_seen.load(Relaxed));
    }
}
