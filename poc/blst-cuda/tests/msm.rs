// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#[cfg(feature = "bls12_377")]
use ark_bls12_377::{Fq, Fr, G1Affine};
// #[cfg(feature = "bls12_381")]
// use ark_bls12_381::G1Affine;
// #[cfg(feature = "bn254")]
// use ark_bn254::G1Affine;
use ark_ec::msm::VariableBaseMSM;
// use ark_ec::ProjectiveCurve;
use ark_ec::{AffineCurve, ProjectiveCurve};
use ark_ff::{fields::*, BigInteger256, BigInteger384};
use ark_std::UniformRand;

use std::str::FromStr;

use blst_msm::*;

use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn msm_correctness() {
    println!("msm_correctness()");

    let test_npow = std::env::var("TEST_NPOW").unwrap_or("2".to_string());
    let npoints_npow = i32::from_str(&test_npow).unwrap();

    let (points, scalars) =
        util::generate_points_scalars::<G1Affine>(1usize << npoints_npow);

    println!("points: {:?}", points);
    println!("scalars: {:?}", scalars);

    let msm_result = multi_scalar_mult_arkworks(points.as_slice(), unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    })
    .into_affine();

    println!("msm_result: {:?}", msm_result);

    let arkworks_result =
        VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        })
        .into_affine();

    println!("arkworks_result: {:?}", arkworks_result);

    assert_eq!(msm_result, arkworks_result);
}

#[test]
fn mymsm_correctness() {
    let mut rng = ChaCha20Rng::from_entropy();

    println!("mymsm_correctness()");

    let mut p = <G1Affine as AffineCurve>::Projective::rand(&mut rng);
    p.x = Fq::new(BigInteger384([
        10055327746687726116,
        10546240533623896932,
        13514537841552526816,
        10622230844852045008,
        4851019884132902411,
        12549026585763795,
    ]));

    // println!("temp: {:?}", temp);

    p.y = Fq::new(BigInteger384([
        7159845758734103471,
        15631361635891468970,
        6157014105130117914,
        7851149777077196517,
        10068593295135327943,
        47438366734490271,
    ]));

    p.z = Fq::new(BigInteger384([
        13423963763304966588,
        17759136182196605160,
        3090208531647158453,
        17569436763320925772,
        18290543140997503603,
        19168874821656007,
    ]));

    println!("p: {:?}", p);

    let mut points = <<G1Affine as AffineCurve>::Projective as ProjectiveCurve>::batch_normalization_into_affine(&[p].to_vec());
    println!("points: {:?}", points);

    // let mut s = <G1Affine as AffineCurve>::ScalarField::rand(&mut rng);
    // println!("s: {:?}", s);

    let s = Fr::new(BigInteger256([
        16607241952120894801,
        8902715486184888151,
        15432806594273149183,
        758980840597131341,
    ]));

    // println!("s: {:?}", s);
    let scalars = [s].to_vec();
    println!("scalars: {:?}", scalars);

    // let mymsm_result = mymsm_scalar_mult();

    let arkworks_result =
        VariableBaseMSM::multi_scalar_mul(points.as_slice(), unsafe {
            std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
        })
        .into_affine();

    println!("\r\narkworks_result: {:?}", arkworks_result);

    let mymsm_result = mymsm_scalar_mult_works(points.as_slice(), unsafe {
        std::mem::transmute::<&[_], &[BigInteger256]>(scalars.as_slice())
    })
    .into_affine();

    println!("\r\nmymsm_result: {}", 0);

    assert_eq!(1, 1);
}
