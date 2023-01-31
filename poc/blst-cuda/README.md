## Test methods
### To test blake2 cuda code

```
// cd poc/blst-cuda
$ cargo test --features=bls12_377 test_polynomial  --  --nocapture

```

### To test blake2 rust code

```
// cd poc/ntt-cuda
$ cargo test --features=bls12_377 test_blake2 -- --nocapture

```
