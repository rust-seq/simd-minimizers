mod dedup;
mod table_lookup;

pub use dedup::append_unique_vals;
pub use packed_seq::intrinsics::transpose;
pub use table_lookup::table_lookup;
