//! Buffer contiguo, alineado a línea de caché, inicializado a cero por el asignador.
//!
//! `alloc_zeroed` para tamaños grandes se traduce en `mmap` anónimo: el kernel entrega
//! páginas COW de la página cero, así que reservar la arena de regrets de todo el árbol es
//! O(1) en tiempo y en RSS; solo se materializan las páginas de infosets realmente visitados
//! (crítico en MCCFR, donde la mayoría de infosets postflop nunca se muestrean).

use std::alloc::{alloc_zeroed, dealloc, handle_alloc_error, Layout};
use std::ops::Deref;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU16, AtomicU32, AtomicU64, AtomicU8};

pub const CACHE_LINE: usize = 64;

/// Tipos para los que el patrón de bits todo-cero es un valor válido.
///
/// # Safety
/// El implementador garantiza que `mem::zeroed::<Self>()` es un valor válido y que el tipo
/// no tiene `Drop` con efectos.
pub unsafe trait ZeroInit: Sized {}
unsafe impl ZeroInit for AtomicU8 {}
unsafe impl ZeroInit for AtomicU16 {}
unsafe impl ZeroInit for AtomicU32 {}
unsafe impl ZeroInit for AtomicU64 {}

pub struct AlignedBuf<T: ZeroInit> {
    ptr: NonNull<T>,
    len: usize,
    layout: Layout,
}

// SAFETY: el buffer es propietario exclusivo de su memoria; compartirlo equivale a compartir `[T]`.
unsafe impl<T: ZeroInit + Send> Send for AlignedBuf<T> {}
unsafe impl<T: ZeroInit + Sync> Sync for AlignedBuf<T> {}

impl<T: ZeroInit> AlignedBuf<T> {
    pub fn zeroed(len: usize) -> Self {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).expect("size overflow").max(1);
        let align = CACHE_LINE.max(std::mem::align_of::<T>());
        let layout = Layout::from_size_align(bytes, align).expect("invalid layout");
        // SAFETY: layout de tamaño > 0; `T: ZeroInit` hace válido el contenido a cero.
        let raw = unsafe { alloc_zeroed(layout) } as *mut T;
        let ptr = NonNull::new(raw).unwrap_or_else(|| handle_alloc_error(layout));
        AlignedBuf { ptr, len, layout }
    }
}

impl<T: ZeroInit> Deref for AlignedBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: `ptr` apunta a `len` elementos válidos durante toda la vida de `self`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: ZeroInit> Drop for AlignedBuf<T> {
    fn drop(&mut self) {
        // SAFETY: mismo layout que en la asignación; `T` no requiere drop (ZeroInit).
        unsafe { dealloc(self.ptr.as_ptr() as *mut u8, self.layout) }
    }
}

impl<T: ZeroInit> std::fmt::Debug for AlignedBuf<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AlignedBuf").field("len", &self.len).field("align", &self.layout.align()).finish()
    }
}
