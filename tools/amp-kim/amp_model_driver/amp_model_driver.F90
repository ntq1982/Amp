!
! CDDL HEADER START
!
! The contents of this file are subject to the terms of the Common Development
! and Distribution License Version 1.0 (the "License").
!
! You can obtain a copy of the license at
! http://www.opensource.org/licenses/CDDL-1.0.  See the License for the
! specific language governing permissions and limitations under the License.
!
! When distributing Covered Code, include this CDDL HEADER in each file and
! include the License file in a prominent location with the name LICENSE.CDDL.
! If applicable, add the following below this CDDL HEADER, with the fields
! enclosed by brackets "[]" replaced with your own identifying information:
!
! Portions Copyright (c) [yyyy] [name of copyright owner]. All rights reserved.
!
! CDDL HEADER END
!
!
! Copyright (c) 2013--2018.
! All rights reserved.
!
! Contributors:
!    Alireza Khorshidi
!    C. Franklin Goldsmith
!    Ryan S. Elliott
!    Malte Doentgen
!    Muammar El-Khatib Rodriguez
!

!****************************************************************************
!**
!**  MODULE amp_model_driver
!**
!**  KIM Model Driver for Atomistic Machine-learning Package (Amp) potentials
!**
!**  Language: Fortran 2003
!**
!****************************************************************************


module amp_model_driver

use neuralnetwork
use, intrinsic :: iso_c_binding
use kim_model_driver_headers_module
implicit none

save
private
public BUFFER_TYPE,               &
       Compute_Energy_Forces,     &
       compute_arguments_create,  &
       compute_arguments_destroy, &
       refresh,                   &
       destroy

! Below are the definitions and values of all Model parameters
integer(c_int), parameter          :: cd = c_double  ! used for literal constants
integer(c_int), parameter          :: DIM=3          ! dimensionality of space

type :: symmetry_function_type
  character(len=8, kind=c_char) :: symmetry_type
  character(len=8, kind=c_char) :: species1_name_string
  type(kim_species_name_type) :: species1_name
  integer(c_int) :: species1_code
  character(len=8, kind=c_char) :: species2_name_string
  type(kim_species_name_type) :: species2_name
  integer(c_int) :: species2_code
  real(c_double) :: eta
  real(c_double) :: gamma
  real(c_double) :: zeta
  real(c_double) :: max_fingerprint
  real(c_double) :: min_fingerprint
end type symmetry_function_type

type :: species_symmetry_function_type
  character(len=8, kind=c_char) :: center_atom_species
  integer(c_int) :: center_atom_species_code
  type(symmetry_function_type), allocatable :: gs(:)
end type species_symmetry_function_type

type :: species_model_type
  integer(c_int) :: no_hiddenlayers_of_species
  integer(c_int), allocatable :: no_nodes_of_species(:)
  character(len=8, kind=c_char) :: activation_function
end type species_model_type

!-------------------------------------------------------------------------------
!
!  Definition of Buffer type
!
!-------------------------------------------------------------------------------
type :: BUFFER_TYPE
  real(c_double) :: influence_distance(1)
  real(c_double) :: cutoff(1)
  integer(c_int) :: padding_neighbor_hints(1)

  integer(c_int) :: num_species
  character(len=8, kind=c_char), allocatable :: species_name_strings(:)
  type(kim_species_name_type), allocatable :: species_names(:)
  integer(c_int), allocatable :: species_codes(:)
  integer(c_int), allocatable :: num_symmetry_functions_of_species(:)
  type(species_symmetry_function_type), allocatable :: symmetry_functions(:)
  type(species_model_type), allocatable :: model_props(:)
  real(c_double), allocatable :: parameters(:)
endtype BUFFER_TYPE

contains

!-------------------------------------------------------------------------------
!
! Compute energy and forces on particles from the positions.
!
!-------------------------------------------------------------------------------
recursive subroutine Compute_Energy_Forces(model_compute_handle, &
  model_compute_arguments_handle, ierr) bind(c)
implicit none

!-- Transferred variables
type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
type(kim_model_compute_arguments_handle_type), intent(in) :: &
  model_compute_arguments_handle
integer(c_int), intent(out) :: ierr

!-- Local variables
integer(c_int) :: ierr2
integer(c_int) :: comp_forces,comp_energy,comp_enepot
type(BUFFER_TYPE), pointer :: buf; type(c_ptr) :: pbuf

!-- KIM variables
real(c_double) :: model_cutoff
integer(c_int), pointer :: num_atoms
real(c_double), pointer :: energy
real(c_double), pointer :: coor(:,:)
real(c_double), pointer :: forces(:,:)
real(c_double), pointer :: enepot(:)
integer(c_int), pointer :: neighbors_of_particle(:)
integer(c_int), pointer :: particle_species_codes(:)
integer(c_int), pointer :: particle_contributing(:)

  integer(c_int) :: index, number_of_neighbors, p, q, count, l, num_gs, symbol
  integer(c_int) :: number_of_neighbor_of_neighbors
  integer(c_int) :: selfindex, selfsymbol, nindex, nsymbol, nnindex, nnsymbol

  integer(c_int), allocatable:: neighbor_numbers(:), neighbor_indices(:)
  real(c_double), allocatable:: neighbor_positions(:,:)
  integer(c_int), allocatable:: neighbor_of_neighbor_numbers(:), neighbor_of_neighbor_indices(:)
  real(c_double), allocatable:: neighbor_of_neighbor_positions(:,:)
  real(c_double), allocatable:: fingerprint(:)
  real(c_double) :: atom_energy
  real(c_double), allocatable:: fingerprintprime(:)
  real(c_double):: dforce
  integer(c_int):: cutofffn_code
  real(c_double):: rc
  real(c_double), dimension(3):: ri

!!!!!!!!!!!!!!!!!!!!!!!!!!! type definition !!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type:: integer_one_d_array
    sequence
    integer(c_int), allocatable:: onedarray(:)
  end type integer_one_d_array

  type:: embedded_real_one_two_d_array
    sequence
    type(real_two_d_array), allocatable:: onedarray(:)
  end type embedded_real_one_two_d_array

  type:: real_one_d_array
    sequence
    real(c_double), allocatable:: onedarray(:)
  end type real_one_d_array

!!!!!!!!!!!!!!!!!!!!!!!!!! dummy variables !!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  type(integer_one_d_array), allocatable :: neighborlists(:)
  type(real_one_d_array), allocatable :: fingerprints(:)
  type(embedded_real_one_two_d_array), allocatable:: fingerprintprimes(:)

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

! get model buffer from KIM object
call kim_get_model_buffer_pointer(model_compute_handle, pbuf)
call c_f_pointer(pbuf, buf)

model_cutoff = buf%influence_distance(1)

! Unpack data from KIM object
!
ierr = 0
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_number_of_particles, num_atoms, ierr2)
ierr = ierr + ierr2

call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_particle_species_codes, &
  num_atoms, particle_species_codes, ierr2)
ierr = ierr + ierr2
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_particle_contributing, num_atoms, particle_contributing, &
  ierr2)
ierr = ierr + ierr2
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_coordinates, dim, num_atoms, coor, ierr2)
ierr = ierr + ierr2
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_partial_energy, energy, ierr2)
ierr = ierr + ierr2
! Force calculation can be turned off by commenting the next four lines.
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_partial_forces, dim, num_atoms, forces, ierr2)
ierr = ierr + ierr2
call kim_get_argument_pointer( &
  model_compute_arguments_handle, &
  kim_compute_argument_name_partial_particle_energy, num_atoms, enepot, ierr2)
ierr = ierr + ierr2
if (ierr /= 0) then
  call kim_log_entry(model_compute_arguments_handle, KIM_LOG_VERBOSITY_ERROR, "get_argument_pointer")
  return
end if

  ! Check to be sure that the species are correct
  do index = 1, num_atoms
    ierr = 1 ! assumes an error
    do p = 1, buf%num_species
      if (particle_species_codes(index) .eq. buf%species_codes(p)) then
        ierr = 0
        exit
      end if
    end do
    if (ierr .ne. 0) then
      call kim_log_entry(model_compute_arguments_handle, KIM_LOG_VERBOSITY_ERROR,  "Unexpected species code detected")
      return
    end if
  end do

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
!  Compute energy and forces (Amp)
!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!
if (associated(energy)) then
  comp_energy =  1
else
  comp_energy = 0
end if
if (associated(forces)) then
  comp_forces = 1
else
  comp_forces = 0
end if
if (associated(enepot)) then
  comp_enepot = 1
else
  comp_enepot = 0
end if

ierr = 0 ! everything is ok

! Initialize potential energies, forces
if (comp_enepot.eq.1) enepot = 0.0_cd
if (comp_energy.eq.1) energy = 0.0_cd
if (comp_forces.eq.1)  forces  = 0.0_cd

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  ! Allocate and set up neighborlists for particles
  allocate(neighborlists(num_atoms))
  do index = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(index) == 1) then
    ierr = 0 ! everything is ok
    call kim_get_neighbor_list( &
    model_compute_arguments_handle, 1, index, number_of_neighbors, &
    neighbors_of_particle, ierr)
    allocate(neighborlists(index)%onedarray(number_of_neighbors))
    do p = 1, number_of_neighbors
      neighborlists(index)%onedarray(p) = neighbors_of_particle(p)
    end do
    if (ierr /= 0) then
      ! some sort of problem, exit
      call kim_log_entry(model_compute_arguments_handle, KIM_LOG_VERBOSITY_ERROR, "kim_api_get_neigh")
      ierr = 1
      return
    end if
  ! end if
  end do

  ! Allocate and set up fingerprints of particles
  allocate(fingerprints(num_atoms))
  rc = buf%cutoff(1)
  cutofffn_code = 1 ! for 'Cosine'
  do index = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(index) == 1) then
    symbol = particle_species_codes(index)
    num_gs = size(buf%symmetry_functions(symbol)%gs)
    ri = coor(:, index)
    number_of_neighbors = size(neighborlists(index)%onedarray)
    allocate(neighbor_numbers(number_of_neighbors))
    allocate(neighbor_positions(number_of_neighbors, 3))
    do p = 1, number_of_neighbors
      nindex = neighborlists(index)%onedarray(p)
      neighbor_numbers(p) = particle_species_codes(nindex)
      neighbor_positions(p, 1) = coor(1, nindex)
      neighbor_positions(p, 2) = coor(2, nindex)
      neighbor_positions(p, 3) = coor(3, nindex)
    end do
    allocate(fingerprint(num_gs))
    call calculate_fingerprint(num_gs, buf%symmetry_functions(symbol)%gs, &
    number_of_neighbors, neighbor_numbers, &
    neighbor_positions, ri, rc, cutofffn_code, fingerprint)
    allocate(fingerprints(index)%onedarray(num_gs))
    do p = 1, num_gs
      fingerprints(index)%onedarray(p) = fingerprint(p)
    end do
    deallocate(neighbor_numbers)
    deallocate(neighbor_positions)
    deallocate(fingerprint)
  ! end if
  end do

if (comp_forces.eq.1) then

  ! Allocate and set up fingerprintprimes of particles
  allocate(fingerprintprimes(num_atoms))
  do selfindex = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(selfindex) == 1) then
    number_of_neighbors = size(neighborlists(selfindex)%onedarray)
    allocate(fingerprintprimes(selfindex)%onedarray(number_of_neighbors))
    do p = 1, number_of_neighbors
      nindex = neighborlists(selfindex)%onedarray(p)
      nsymbol = particle_species_codes(nindex)
      num_gs = size(buf%symmetry_functions(nsymbol)%gs)
      allocate(fingerprintprimes(selfindex)%onedarray(p)%twodarray(3, num_gs))

      number_of_neighbor_of_neighbors = size(neighborlists(nindex)%onedarray)
      allocate(neighbor_of_neighbor_numbers(number_of_neighbor_of_neighbors))
      allocate(neighbor_of_neighbor_indices(number_of_neighbor_of_neighbors))
      allocate(neighbor_of_neighbor_positions(number_of_neighbor_of_neighbors, 3))
      do q = 1, number_of_neighbor_of_neighbors
        nnindex = neighborlists(nindex)%onedarray(q)
        nnsymbol = particle_species_codes(nnindex)
        neighbor_of_neighbor_indices(q) = nnindex
        neighbor_of_neighbor_numbers(q) = nnsymbol
        neighbor_of_neighbor_positions(q, 1) = coor(1, nnindex)
        neighbor_of_neighbor_positions(q, 2) = coor(2, nnindex)
        neighbor_of_neighbor_positions(q, 3) = coor(3, nnindex)
      end do
      do l = 0, 2
        allocate(fingerprintprime(num_gs))
        call calculate_fingerprintprime(num_gs, buf%symmetry_functions(nsymbol)%gs, &
        number_of_neighbor_of_neighbors, neighbor_of_neighbor_indices, neighbor_of_neighbor_numbers, &
        neighbor_of_neighbor_positions, rc, cutofffn_code, nindex, coor(:, nindex), selfindex, l, fingerprintprime)
        do q = 1, num_gs
          fingerprintprimes(selfindex)%onedarray(p)%twodarray(l+1, q) = fingerprintprime(q)
        end do
        deallocate(fingerprintprime)
      end do
      deallocate(neighbor_of_neighbor_numbers)
      deallocate(neighbor_of_neighbor_indices)
      deallocate(neighbor_of_neighbor_positions)
    end do
  ! end if
  end do

end if

  ! As of now, the code only works if the number of fingerprints for different species are the same.
  allocate(min_fingerprints(buf%num_species, size(buf%symmetry_functions(1)%gs)))
  allocate(max_fingerprints(buf%num_species, size(buf%symmetry_functions(1)%gs)))
  do p = 1, buf%num_species
    do q = 1, size(buf%symmetry_functions(p)%gs)
      min_fingerprints(p, q) = buf%symmetry_functions(p)%gs(q)%min_fingerprint
      max_fingerprints(p, q) = buf%symmetry_functions(p)%gs(q)%max_fingerprint
    end do
  end do
  allocate(no_layers_of_elements(buf%num_species))
  do p = 1, buf%num_species
    no_layers_of_elements(p) = buf%model_props(p)%no_hiddenlayers_of_species + 2
  end do
  count = 0
  do p = 1, buf%num_species
    count = count + buf%model_props(p)%no_hiddenlayers_of_species + 2
  end do
  allocate(no_nodes_of_elements(count))
  count = 1
  do p = 1, buf%num_species
    no_nodes_of_elements(count) = size(buf%symmetry_functions(p)%gs)
    count = count + 1
    do q = 1, buf%model_props(p)%no_hiddenlayers_of_species
      no_nodes_of_elements(count) = buf%model_props(p)%no_nodes_of_species(q)
      count = count + 1
    end do
    no_nodes_of_elements(count) = 1
    count = count + 1
  end do
  ! As of now, activation function should be the same for different species neural nets.
  if (buf%model_props(1)%activation_function == 'tanh') then
    activation_signal = 1
  else if (buf%model_props(1)%activation_function == 'sigmoid') then
    activation_signal = 2
  else if (buf%model_props(1)%activation_function == 'linear') then
    activation_signal = 3
  end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

if (comp_enepot.eq.1 .OR. comp_energy.eq.1) then

  ! Initialize energy
  energy = 0.0_cd
  !  Loop over particles and compute energy
  do selfindex = 1, num_atoms
   ! This is related to the ghost atoms, not checked in this version of code.
   ! if (particle_contributing(selfindex) == 1) then
      selfsymbol = particle_species_codes(selfindex)
      atom_energy = calculate_atomic_energy(selfsymbol, &
      size(buf%symmetry_functions(selfsymbol)%gs), &
      fingerprints(selfindex)%onedarray, buf%num_species, &
      buf%species_codes, size(buf%parameters), buf%parameters)
      energy = energy + atom_energy
!      enepot(selfindex) = atom_energy
   ! end if
  end do

end if

if (comp_forces.eq.1) then

  ! Initialize forces
  do selfindex = 1, num_atoms
    do p = 1, 3
      forces(p, selfindex) = 0.0d0
    end do
  end do
  !  Loop over particles and their neighbors and compute forces
  do selfindex = 1, num_atoms
   ! This is related to the ghost atoms, not checked in this version of code.
   ! if (particle_contributing(selfindex) == 1) then

      ! First the contribution of self particle on itself is calculated for forces
      selfsymbol = particle_species_codes(selfindex)
      num_gs = size(buf%symmetry_functions(selfsymbol)%gs)
      number_of_neighbors = size(neighborlists(selfindex)%onedarray)
      allocate(neighbor_numbers(number_of_neighbors))
      allocate(neighbor_indices(number_of_neighbors))
      allocate(neighbor_positions(number_of_neighbors, 3))
      do p = 1, number_of_neighbors
        nindex = neighborlists(selfindex)%onedarray(p)
        neighbor_indices(p) = nindex
        neighbor_numbers(p) = particle_species_codes(nindex)
        neighbor_positions(p, 1) = coor(1, nindex)
        neighbor_positions(p, 2) = coor(2, nindex)
        neighbor_positions(p, 3) = coor(3, nindex)
      end do
      allocate(fingerprint(num_gs))
      do q = 1, num_gs
        fingerprint(q) = fingerprints(selfindex)%onedarray(q)
      end do
      do l = 0, 2
        allocate(fingerprintprime(num_gs))
        call calculate_fingerprintprime(num_gs, buf%symmetry_functions(selfsymbol)%gs, &
        number_of_neighbors, neighbor_indices, neighbor_numbers, &
        neighbor_positions, rc, cutofffn_code, selfindex, coor(:, selfindex), selfindex, l, &
        fingerprintprime)
        dforce = calculate_force(selfsymbol, num_gs, &
        fingerprint, fingerprintprime, &
        buf%num_species, buf%species_codes, &
        size(buf%parameters), buf%parameters)
        forces(l + 1, selfindex) = forces(l + 1, selfindex) + dforce
        deallocate(fingerprintprime)
      end do
      deallocate(fingerprint)
      deallocate(neighbor_numbers)
      deallocate(neighbor_indices)
      deallocate(neighbor_positions)

      ! Second the contribution of neighbors on self particle is calculated for forces
      do p = 1, number_of_neighbors
        nindex = neighborlists(selfindex)%onedarray(p)
        nsymbol = particle_species_codes(nindex)
        num_gs = size(buf%symmetry_functions(nsymbol)%gs)
        allocate(fingerprint(num_gs))
        do q = 1, num_gs
          fingerprint(q) = fingerprints(nindex)%onedarray(q)
        end do
        do l = 0, 2
          allocate(fingerprintprime(num_gs))
          do q = 1, num_gs
            fingerprintprime(q) = &
            fingerprintprimes(selfindex)%onedarray(p)%twodarray(l + 1, q)
          end do
          dforce = calculate_force(nsymbol, num_gs, &
          fingerprint, fingerprintprime, &
          buf%num_species, buf%species_codes, &
          size(buf%parameters), buf%parameters)
          forces(l + 1, selfindex) = forces(l + 1, selfindex) + dforce
          deallocate(fingerprintprime)
        end do
        deallocate(fingerprint)
      end do
   ! end if
  end do

end if

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  deallocate(min_fingerprints)
  deallocate(max_fingerprints)
  deallocate(no_layers_of_elements)
  deallocate(no_nodes_of_elements)

if (comp_forces.eq.1) then
  ! Deallocate fingerprintprimes of particles
  do selfindex = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(selfindex) == 1) then
    number_of_neighbors = size(neighborlists(selfindex)%onedarray)
    do p = 1, number_of_neighbors
      deallocate(fingerprintprimes(selfindex)%onedarray(p)%twodarray)
    end do
    deallocate(fingerprintprimes(selfindex)%onedarray)
  ! end if
  end do
  deallocate(fingerprintprimes)
end if

  ! Deallocate fingerprints of particles
  do index = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(index) == 1) then
    deallocate(fingerprints(index)%onedarray)
  ! end if
  end do
  deallocate(fingerprints)

  ! Deallocate neighborlist of particles
  do index = 1, num_atoms
  ! This is related to the ghost atoms, not checked in this version of code.
  ! if (particle_contributing(index) == 1) then
    deallocate(neighborlists(index)%onedarray)
  ! end if
  end do
  deallocate(neighborlists)

ierr = 0  ! Everything is great
return

end subroutine Compute_Energy_Forces


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

recursive subroutine calculate_fingerprintprime(num_gs, gs, number_of_neighbors, neighbor_indices, neighbor_numbers, &
neighbor_positions, rc, cutofffn_code, i, ri, m, l, fingerprintprime)

implicit none

integer(c_int):: num_gs, number_of_neighbors, i, m, l, q
type(symmetry_function_type) :: gs(num_gs), g
integer(c_int), dimension(number_of_neighbors):: neighbor_numbers, neighbor_indices
real(c_double), dimension(number_of_neighbors, 3):: neighbor_positions
real(c_double), dimension(3):: ri
real(c_double)::  rc, ridge
integer(c_int):: cutofffn_code
real(c_double):: fingerprintprime(num_gs)
integer(c_int), dimension(2):: g_numbers

do q = 1, num_gs
  g = gs(q)
  if (g%symmetry_type == 'g2') then
    call calculate_g2_prime(neighbor_indices, &
    neighbor_numbers, neighbor_positions, g%species1_code, &
    g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g4') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g4_prime(neighbor_indices, neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g5') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g5_prime(neighbor_indices, neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, i, ri, m, l, number_of_neighbors, ridge)
  else
    print *, "Unknown symmetry function type! Only 'g2', 'g4', and 'g5' are supported."
  end if
  fingerprintprime(q) = ridge
end do

end subroutine calculate_fingerprintprime


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

recursive subroutine calculate_fingerprint(num_gs, gs, number_of_neighbors, neighbor_numbers, &
neighbor_positions, ri, rc, cutofffn_code, fingerprint)

implicit none

integer(c_int):: num_gs, number_of_neighbors, q
type(symmetry_function_type) :: gs(num_gs), g
integer(c_int), dimension(number_of_neighbors):: neighbor_numbers
real(c_double), dimension(number_of_neighbors, 3):: neighbor_positions
real(c_double), dimension(3):: ri
real(c_double)::  rc, ridge
integer(c_int):: cutofffn_code
real(c_double), dimension(num_gs):: fingerprint
integer(c_int), dimension(2):: g_numbers

do q = 1, num_gs
  g = gs(q)
  if (g%symmetry_type == 'g2') then
    call calculate_g2(neighbor_numbers, neighbor_positions, &
    g%species1_code, g%eta, g%gamma, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g4') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g4(neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else if (g%symmetry_type == 'g5') then
    g_numbers(1) = g%species1_code
    g_numbers(2) = g%species2_code
    call calculate_g5(neighbor_numbers, neighbor_positions, &
    g_numbers, g%gamma, g%zeta, g%eta, rc, cutofffn_code, ri, number_of_neighbors, ridge)
  else
    print *, "Unknown symmetry function type! Only 'g2', 'g4', and 'g5' are supported."
  end if
  fingerprint(q) = ridge
end do

end subroutine calculate_fingerprint

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


!-------------------------------------------------------------------------------
!
! Model driver refresh routine
!
!-------------------------------------------------------------------------------
recursive subroutine refresh(model_refresh_handle, ierr) bind(c)
implicit none

!-- Transferred variables
type(kim_model_refresh_handle_type), intent(inout) :: model_refresh_handle
integer(c_int), intent(out) :: ierr

!-- Local variables
real(c_double) energy_at_cutoff
type(BUFFER_TYPE), pointer :: buf; type(c_ptr) :: pbuf

! get model buffer from KIM object
call kim_get_model_buffer_pointer(model_refresh_handle, pbuf)
call c_f_pointer(pbuf, buf)

call kim_set_influence_distance_pointer(model_refresh_handle, &
  buf%influence_distance(1))
call kim_set_neighbor_list_pointers(model_refresh_handle, &
  1, buf%influence_distance, buf%padding_neighbor_hints)

! Set new values in KIM object and buffer
buf%influence_distance(1) = buf%cutoff(1)

ierr = 0
return

end subroutine refresh

!-------------------------------------------------------------------------------
!
! Model driver destroy routine
!
!-------------------------------------------------------------------------------
recursive subroutine destroy(model_destroy_handle, ierr) bind(c)
implicit none

!-- Transferred variables
type(kim_model_destroy_handle_type), intent(inout) :: model_destroy_handle
integer(c_int), intent(out) :: ierr

!-- Local variables
type(BUFFER_TYPE), pointer :: buf; type(c_ptr) :: pbuf

! get model buffer from KIM object
call kim_get_model_buffer_pointer(model_destroy_handle, pbuf)
call c_f_pointer(pbuf, buf)

deallocate( buf )

ierr = 0
return

end subroutine destroy

!-------------------------------------------------------------------------------
!
! Model driver compute arguments create routine
!
!-------------------------------------------------------------------------------
recursive subroutine compute_arguments_create(model_compute_handle, &
  model_compute_arguments_create_handle, ierr) bind(c)
implicit none

!-- Transferred variables
type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
type(kim_model_compute_arguments_create_handle_type), intent(inout) :: &
  model_compute_arguments_create_handle
integer(c_int), intent(out) :: ierr

integer(c_int) ierr2

ierr = 0
ierr2 = 0

! register arguments
call kim_set_argument_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_argument_name_partial_energy, &
  kim_support_status_optional, ierr)
call kim_set_argument_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_argument_name_partial_forces, &
  kim_support_status_optional, ierr2)
ierr = ierr + ierr2
call kim_set_argument_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_argument_name_partial_particle_energy, &
  kim_support_status_optional, ierr2)
ierr = ierr + ierr2
if (ierr /= 0) then
  call kim_log_entry(model_compute_arguments_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to register arguments support_status")
  goto 42
end if

! register callbacks
call kim_set_callback_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_callback_name_process_dedr_term, &
  kim_support_status_optional, ierr)
call kim_set_callback_support_status( &
  model_compute_arguments_create_handle, &
  kim_compute_callback_name_process_d2edr2_term, &
  kim_support_status_optional, ierr2)
ierr = ierr + ierr2
if (ierr /= 0) then
  call kim_log_entry(model_compute_arguments_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to register callbacks support_status")
  goto 42
end if

ierr = 0
42 continue
return

end subroutine compute_arguments_create

!-------------------------------------------------------------------------------
!
! Model driver compute arguments destroy routine
!
!-------------------------------------------------------------------------------
recursive subroutine compute_arguments_destroy(model_compute_handle, &
  model_compute_arguments_destroy_handle, ierr) bind(c)
implicit none

!-- Transferred variables
type(kim_model_compute_handle_type), intent(in) :: model_compute_handle
type(kim_model_compute_arguments_destroy_handle_type), intent(inout) :: &
  model_compute_arguments_destroy_handle
integer(c_int), intent(out) :: ierr

! nothing to be done

ierr = 0

return
end subroutine compute_arguments_destroy

end module amp_model_driver

!-------------------------------------------------------------------------------
!
! Model driver create routine (REQUIRED)
!
!-------------------------------------------------------------------------------
recursive subroutine model_driver_create_routine(model_driver_create_handle, &
  requested_length_unit, requested_energy_unit, requested_charge_unit, &
  requested_temperature_unit, requested_time_unit, ierr) bind(c)
use, intrinsic :: iso_c_binding
use amp_model_driver
use kim_model_driver_headers_module
implicit none
integer(c_int), parameter :: cd = c_double ! used for literal constants

!-- Transferred variables
type(kim_model_driver_create_handle_type), intent(inout) &
  :: model_driver_create_handle
type(kim_length_unit_type), intent(in), value :: requested_length_unit
type(kim_energy_unit_type), intent(in), value :: requested_energy_unit
type(kim_charge_unit_type), intent(in), value :: requested_charge_unit
type(kim_temperature_unit_type), intent(in), value :: requested_temperature_unit
type(kim_time_unit_type), intent(in), value :: requested_time_unit
integer(c_int), intent(out) :: ierr

!-- Local variables
integer(c_int) :: number_of_parameter_files
character(len=1024, kind=c_char) :: parameter_file_name
integer(c_int) :: ierr2
type(BUFFER_TYPE), pointer :: buf

integer(c_int) :: p, q, k, count
character(len=8, kind=c_char) :: junk

! register units
! use requested units (we'll convert parameters as needed below).
! We only make use of length and energy, other are unused.
call kim_set_units( &
  model_driver_create_handle, &
  requested_length_unit, &
  requested_energy_unit, &
  kim_charge_unit_unused, &
  kim_temperature_unit_unused, &
  kim_time_unit_unused, ierr)
if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to set units")
  goto 42
end if

! register numbering
! we'll use one-based numbering.
call kim_set_model_numbering( &
  model_driver_create_handle, kim_numbering_one_based, ierr)
if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to set numbering")
  goto 42
end if

! store callback pointers in KIM object
call kim_set_routine_pointer( &
  model_driver_create_handle, &
  KIM_MODEL_ROUTINE_NAME_COMPUTE, &   
  KIM_LANGUAGE_NAME_FORTRAN, 1, &
  c_funloc(Compute_Energy_Forces), ierr)
call kim_set_routine_pointer( &
  model_driver_create_handle, &
  KIM_MODEL_ROUTINE_NAME_COMPUTE_ARGUMENTS_CREATE, &
  kim_language_name_fortran, 1, &
  c_funloc(compute_arguments_create), ierr2)
ierr = ierr + ierr2
call kim_set_routine_pointer( &
  model_driver_create_handle, &
  KIM_MODEL_ROUTINE_NAME_COMPUTE_ARGUMENTS_DESTROY, &
  KIM_LANGUAGE_NAME_FORTRAN, 1, &
  c_funloc(compute_arguments_destroy), ierr2)
ierr = ierr + ierr2
call kim_set_routine_pointer( &
  model_driver_create_handle, &
  KIM_MODEL_ROUTINE_NAME_REFRESH, &
  KIM_LANGUAGE_NAME_FORTRAN, 1, &
  c_funloc(refresh), ierr2)
ierr = ierr + ierr2
call kim_set_routine_pointer( &
  model_driver_create_handle, &
  KIM_MODEL_ROUTINE_NAME_DESTROY, &
  KIM_LANGUAGE_NAME_FORTRAN, 1, &
  c_funloc(destroy), ierr2)
ierr = ierr + ierr2
if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to store callback pointers")
  ierr = 1
  goto 42
end if

! process parameter files
call kim_get_number_of_parameter_files( &
  model_driver_create_handle, number_of_parameter_files)
if (number_of_parameter_files .ne. 1) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Wrong number of parameter files")
  ierr = 1
  goto 42
end if

! allocate model_buffer object and register it in the model_drier_create object
allocate(buf)
call kim_set_model_buffer_pointer( &
  model_driver_create_handle, c_loc(buf))

! Read in model parameters from parameter file
call kim_get_parameter_file_name( &
  model_driver_create_handle, 1, parameter_file_name, ierr)
if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to get parameter file name")
  ierr = 1
  goto 42
end if
open(10,file=parameter_file_name,status="old")
read(10,*,iostat=ierr,err=100) buf%num_species
allocate(buf%species_name_strings(buf%num_species))
read(10,*,iostat=ierr,err=100) (buf%species_name_strings(p), p = 1, buf%num_species)
allocate(buf%num_symmetry_functions_of_species(buf%num_species))
read(10,*,iostat=ierr,err=100) (buf%num_symmetry_functions_of_species(p), p = 1, buf%num_species)
allocate(buf%symmetry_functions(buf%num_species))
do p = 1, buf%num_species
  buf%symmetry_functions(p)%center_atom_species_code = p
  buf%symmetry_functions(p)%center_atom_species = buf%species_name_strings(p)
  allocate(buf%symmetry_functions(p)%gs(buf%num_symmetry_functions_of_species(p)))
  do q = 1, buf%num_symmetry_functions_of_species(p)
    read(10,*,iostat=ierr,err=100) &
    junk, &
    buf%symmetry_functions(p)%gs(q)%symmetry_type
    if (buf%symmetry_functions(p)%gs(q)%symmetry_type == 'g2') then
      read(10,*,iostat=ierr,err=100) &
      buf%symmetry_functions(p)%gs(q)%species1_name_string, &
      buf%symmetry_functions(p)%gs(q)%eta
      do k = 1, buf%num_species
        if (buf%species_name_strings(k) == buf%symmetry_functions(p)%gs(q)%species1_name_string) then
          exit
        end if
      end do
      buf%symmetry_functions(p)%gs(q)%species1_code = k
      call kim_from_string(trim(buf%symmetry_functions(p)%gs(q)%species1_name_string), &
      buf%symmetry_functions(p)%gs(q)%species1_name)
      call kim_set_species_code( &
      model_driver_create_handle, buf%symmetry_functions(p)%gs(q)%species1_name, &
      buf%symmetry_functions(p)%gs(q)%species1_code, ierr)
    else if (buf%symmetry_functions(p)%gs(q)%symmetry_type == 'g4') then
      read(10,*,iostat=ierr,err=100) &
      buf%symmetry_functions(p)%gs(q)%species1_name_string, &
      buf%symmetry_functions(p)%gs(q)%species2_name_string, &
      buf%symmetry_functions(p)%gs(q)%eta, &
      buf%symmetry_functions(p)%gs(q)%gamma, &
      buf%symmetry_functions(p)%gs(q)%zeta
      do k = 1, buf%num_species
        if (buf%species_name_strings(k) == buf%symmetry_functions(p)%gs(q)%species1_name_string) then
          exit
        end if
      end do
      buf%symmetry_functions(p)%gs(q)%species1_code = k
      call kim_from_string(trim(buf%symmetry_functions(p)%gs(q)%species1_name_string), &
      buf%symmetry_functions(p)%gs(q)%species1_name)
      call kim_set_species_code( &
      model_driver_create_handle, buf%symmetry_functions(p)%gs(q)%species1_name, &
      buf%symmetry_functions(p)%gs(q)%species1_code, ierr)
      do k = 1, buf%num_species
        if (buf%species_name_strings(k) == buf%symmetry_functions(p)%gs(q)%species2_name_string) then
          exit
        end if
      end do
      buf%symmetry_functions(p)%gs(q)%species2_code = k
      call kim_from_string(trim(buf%symmetry_functions(p)%gs(q)%species2_name_string), &
      buf%symmetry_functions(p)%gs(q)%species2_name)
      call kim_set_species_code( &
      model_driver_create_handle, buf%symmetry_functions(p)%gs(q)%species2_name, &
      buf%symmetry_functions(p)%gs(q)%species2_code, ierr)
    end if
    read(10,*,iostat=ierr,err=100) buf%symmetry_functions(p)%gs(q)%min_fingerprint, &
    buf%symmetry_functions(p)%gs(q)%max_fingerprint
  end do
end do
read(10,*,iostat=ierr,err=100) buf%cutoff(1)  ! in A
allocate(buf%model_props(buf%num_species))
read(10,*,iostat=ierr,err=100) buf%model_props(1)%activation_function
do p = 2, buf%num_species
  buf%model_props(p)%activation_function = buf%model_props(1)%activation_function
end do

do p = 1, buf%num_species
  read(10,*,iostat=ierr,err=100) buf%model_props(p)%no_hiddenlayers_of_species
  allocate(buf%model_props(p)%no_nodes_of_species(buf%model_props(p)%no_hiddenlayers_of_species))
  read(10,*,iostat=ierr,err=100) &
  (buf%model_props(p)%no_nodes_of_species(q), q = 1, buf%model_props(p)%no_hiddenlayers_of_species)
end do

count = 0
do p = 1, buf%num_species
  if (buf%model_props(p)%no_hiddenlayers_of_species == 0) then
    count = count + (size(buf%symmetry_functions(p)%gs) + 1)
  else
    count = count + (size(buf%symmetry_functions(p)%gs) + 1) * buf%model_props(p)%no_nodes_of_species(1)
    do q = 1, buf%model_props(p)%no_hiddenlayers_of_species - 1
      count = count + (buf%model_props(p)%no_nodes_of_species(q) + 1) * buf%model_props(p)%no_nodes_of_species(q + 1)
    end do
    count = count + (buf%model_props(p)%no_nodes_of_species(buf%model_props(p)%no_hiddenlayers_of_species) + 1)
  end if
  count = count + 2
end do
allocate(buf%parameters(count))
read(10,*,iostat=ierr,err=100) (buf%parameters(p), p = 1, count)

close(10)

goto 200
100 continue
! reading parameters failed
ierr = 1
call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to read Amp parameters")
goto 42

200 continue


! register species
allocate(buf%species_names(buf%num_species))
allocate(buf%species_codes(buf%num_species))
do p = 1, buf%num_species
  call kim_from_string(trim(buf%species_name_strings(p)), &
  buf%species_names(p))
  buf%species_codes(p) = p
  call kim_set_species_code( &
  model_driver_create_handle, buf%species_names(p), buf%species_codes(p), ierr)
end do

if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "Unable to set species_names or species_codes")
  goto 42
end if


buf%influence_distance(1) = buf%cutoff(1)
buf%padding_neighbor_hints = 1

! store model cutoff in KIM object
call kim_set_influence_distance_pointer( &
  model_driver_create_handle, buf%influence_distance(1))
call kim_set_neighbor_list_pointers( &
  model_driver_create_handle, 1, buf%influence_distance, &
  buf%padding_neighbor_hints)
! end setup buffer

! store in model buffer
call kim_set_model_buffer_pointer( &
  model_driver_create_handle, c_loc(buf))

! set pointers to parameters in KIM object
call kim_set_parameter_pointer( &
     model_driver_create_handle, buf%cutoff, "cutoff", &
     "Cutoff distance of the model", ierr)
if (ierr /= 0) then
  call kim_log_entry(model_driver_create_handle, KIM_LOG_VERBOSITY_ERROR, "set_parameter")
   goto 42
end if


ierr = 0
42 continue
return

end subroutine model_driver_create_routine