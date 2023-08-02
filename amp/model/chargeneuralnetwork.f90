!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module that utilizes the regression model to calculate energies
!     and forces as well as their derivatives. Function names ending
!     with an underscore correspond to image-centered mode.

      module chargeneuralnetwork
      use neuralnetwork
      implicit none

!     the data of chargeneuralnetwork (should be fed in by python)
      double precision, allocatable::en_min_fingerprints(:, :)
      double precision, allocatable::en_max_fingerprints(:, :)
      double precision, allocatable::charge_min_fingerprints(:, :)
      double precision, allocatable::charge_max_fingerprints(:, :)
      integer, allocatable:: en_no_layers_of_elements(:)
      integer, allocatable:: en_no_nodes_of_elements(:)
      integer:: en_activation_signal
      integer, allocatable:: charge_no_layers_of_elements(:)
      integer, allocatable:: charge_no_nodes_of_elements(:)
      integer:: charge_activation_signal

      !type:: real_two_d_array
      !  sequence
      !  double precision, allocatable:: twodarray(:,:)
      !end type real_two_d_array

      !type:: element_parameters
      !  sequence
      !  double precision:: intercept
      !  double precision:: slope
      !  type(real_two_d_array), allocatable:: weights(:)
      !end type element_parameters

      type:: element_parameters_four
        sequence
        double precision:: en_intercept
        double precision:: en_slope
        double precision:: ei
        double precision:: jii
        type(real_two_d_array), allocatable:: en_weights(:)
      end type element_parameters_four

      type:: element_parameters_six
        sequence
        double precision:: en_intercept
        double precision:: en_slope
        double precision:: intercept
        double precision:: slope
        double precision:: ei
        double precision:: jii
        type(real_two_d_array), allocatable:: en_weights(:)
        type(real_two_d_array), allocatable:: weights(:)
      end type element_parameters_six

      !type:: real_one_d_array
      !  sequence
      !  double precision, allocatable:: onedarray(:)
      !end type real_one_d_array

      contains


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the atom-centered mode.
      function calculate_atomic_charge(symbol, &
      len_of_charge_fingerprint, charge_fingerprint, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, num_parameters, &
      num_elements, &
      len_of_charge_fingerprint
      double precision:: charge_fingerprint(len_of_charge_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_atomic_charge

      integer:: p, element, m, n, layer
      integer:: k, l, j, num_rows, num_cols, q
      integer, allocatable:: charge_hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(element_parameters):: unraveled_parameters(num_elements)
      double precision:: charge_fingerprint_(len_of_charge_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            charge_fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (charge_fingerprint(l) - charge_min_fingerprints(element, l)) / &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprint_(l) = charge_fingerprint(l)
        end if
      end do

!     changing the form of parameters from vector into derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        if (element .GT. 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            l = l + num_rows * num_cols
        end do
      end do
      k = 0
      l = l + 2 * num_elements
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_parameters(element)%weights(j)%twodarray(&
                    p, q) = parameters(l + (p - 1) * num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_parameters(element)%intercept = &
        parameters(l + 2 *  element - 1)
        unraveled_parameters(element)%slope = &
        parameters(l + 2 * element)
      end do

      p = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p = p + charge_no_layers_of_elements(element)
          end if
      end do
      allocate(charge_hiddensizes(charge_no_layers_of_elements(element) - 2))
      do m = 1, charge_no_layers_of_elements(element) - 2
          charge_hiddensizes(m) = charge_no_nodes_of_elements(p + m + 1)
      end do

      allocate(o(charge_no_layers_of_elements(element)))
      allocate(ohat(charge_no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_charge_fingerprint))
      allocate(ohat(1)%onedarray(len_of_charge_fingerprint + 1))
      do m = 1, (len_of_charge_fingerprint)
          o(1)%onedarray(m) = charge_fingerprint_(m)
      end do
      do layer = 1, size(charge_hiddensizes) + 1
          do m = 1, size(unraveled_parameters(element)%weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_parameters(element)%weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_parameters(element)%weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_parameters(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (charge_activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (charge_activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (charge_activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      calculate_atomic_charge = unraveled_parameters(element)%slope * &
      o(layer)%onedarray(1) + unraveled_parameters(element)%intercept
      !calculate_atomic_charge = 1.0d0

!     deallocating neural network
      deallocate(charge_hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)

!      deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
          num_rows = charge_no_nodes_of_elements(k + j) + 1
          num_cols = charge_no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%weights)
      end do

      end function calculate_atomic_charge

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the atom-centered mode.
      function calculate_atomic_electronegtivity(symbol, atomiccharge,&
      len_of_fingerprint, fingerprint, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, num_parameters, &
      len_of_fingerprint, num_elements
      double precision:: fingerprint(len_of_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_atomic_electronegtivity
      double precision:: calculate_atomic_en
      double precision:: atomiccharge
      double precision:: cae1, cae2

      integer:: p, element, m, n, layer
      integer:: k, l, j, num_rows, num_cols, q
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(element_parameters_four):: unraveled_parameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, len_of_fingerprint
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - en_min_fingerprints(element, l)) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

!     changing the form of parameters from vector into derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_parameters(element)%en_weights(j)%twodarray(&
                    p, q) = parameters(l + (p - 1) * num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_parameters(element)%en_intercept = &
        parameters(l + 2 * element - 1)
        unraveled_parameters(element)%en_slope = &
        parameters(l + 2 * element )
      end do
      l = l + 2 * num_elements
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            l = l + num_rows * num_cols
        end do
      end do
      l = l + 2 * num_elements
      
      do element = 1, num_elements
        unraveled_parameters(element)%ei = &
        parameters(l + element)
        unraveled_parameters(element)%jii = &
        parameters(l + num_elements + element)
      end do

      p = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p = p + en_no_layers_of_elements(element)
          end if
      end do
      allocate(hiddensizes(en_no_layers_of_elements(element) - 2))
      do m = 1, en_no_layers_of_elements(element) - 2
          hiddensizes(m) = en_no_nodes_of_elements(p + m + 1)
      end do

      allocate(o(en_no_layers_of_elements(element)))
      allocate(ohat(en_no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_parameters(element)%en_weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(n, m)
              end do
              if (en_activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (en_activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (en_activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      calculate_atomic_en = unraveled_parameters(element)%en_slope * &
      o(layer)%onedarray(1) + unraveled_parameters(element)%en_intercept

      cae1 = calculate_atomic_en * atomiccharge
      cae2 = (1.0d0 / 2.0d0) * unraveled_parameters(element)%jii * &
      (atomiccharge ** 2.0d0)

      calculate_atomic_electronegtivity = &
              unraveled_parameters(element)%ei + &
              cae1 + cae2
!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)

!      deallocating derived type parameters
      do element = 1, num_elements
        deallocate(unraveled_parameters(element)%en_weights)
      end do

      end function calculate_atomic_electronegtivity


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns force value in the atom-centered mode.
      function calculate_atomic_gc_force(symbol, &
      len_of_fingerprint, fingerprint, &
      len_of_charge_fingerprint, charge_fingerprint, &
      fingerprintprime, charge_fingerprintprime,&
      work_function, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, len_of_fingerprint, &
      len_of_charge_fingerprint, num_parameters
      integer:: num_elements
      double precision:: fingerprint(len_of_fingerprint)
      double precision:: charge_fingerprint(len_of_charge_fingerprint)
      double precision:: fingerprintprime(len_of_fingerprint)
      double precision:: charge_fingerprintprime(len_of_charge_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_atomic_gc_force
      double precision:: force_atomiccharge
      double precision:: work_function
      double precision:: force_atomicchi
      double precision:: calculate_force_qi
      double precision:: calculate_force_chi

      double precision, allocatable:: temp(:)
      integer:: p, q, element, m, n, nn, layer, p_charge
      integer:: k, l, j, num_rows, num_cols
      integer, allocatable:: hiddensizes(:)
      integer, allocatable:: charge_hiddensizes(:)
      double precision, allocatable:: net(:)
      double precision, allocatable:: charge_net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: doutputs_dinputs(:)
      type(real_one_d_array), allocatable:: charge_o(:), charge_ohat(:)
      type(real_one_d_array), allocatable:: charge_doutputs_dinputs(:)
      type(element_parameters_four):: unraveled_parameters(num_elements)
      type(element_parameters):: unraveled_charge_parameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)
      double precision:: fingerprintprime_(len_of_fingerprint)
      double precision:: charge_fingerprint_(len_of_charge_fingerprint)
      double precision:: & 
      charge_fingerprintprime_(len_of_charge_fingerprint)


      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do

      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            charge_fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (charge_fingerprint(l) - charge_min_fingerprints(element,l)) / &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprint_(l) = charge_fingerprint(l)
        endif
      end do

      do l = 1, (len_of_fingerprint)
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - en_min_fingerprints(element,l)) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

      ! scaling fingerprintprimes
      do l = 1, len_of_fingerprint
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprintprime_(l) = &
            2.0d0 * fingerprintprime(l) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprintprime_(l) = fingerprintprime(l)
        endif
      end do

      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            charge_fingerprintprime_(l) = &
            2.0d0 * charge_fingerprintprime(l) / &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprintprime_(l) = charge_fingerprintprime(l)
        endif
      end do

!     changing the form of parameters to derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_parameters(element)%en_weights(j)%twodarray(&
                    p, q) = parameters(l + (p - 1) * num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_parameters(element)%en_intercept = &
        parameters(l + 2 * element - 1)
        unraveled_parameters(element)%en_slope = &
        parameters(l + 2 * element )
      end do
      l = l + 2 * num_elements

      k = 0
      do element = 1, num_elements
        allocate(unraveled_charge_parameters(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_charge_parameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_charge_parameters(element)%weights(&
                    j)%twodarray(p, q)  = parameters(l + (p - 1) * &
                    num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_charge_parameters(element)%intercept = &
        parameters(l + 2 *  element - 1)
        unraveled_charge_parameters(element)%slope = &
        parameters(l + 2 * element)
      end do
      l = l + 2 * num_elements


      do element = 1, num_elements
        unraveled_parameters(element)%ei = &
        parameters(l + element)
        unraveled_parameters(element)%jii = &
        parameters(l + num_elements + element)
      end do


      p = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p = p + en_no_layers_of_elements(element)
          end if
      end do

      p_charge = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then 
              exit 
          else 
              p_charge = p_charge + charge_no_layers_of_elements(element)
          end if
      end do



      allocate(hiddensizes(en_no_layers_of_elements(element) - 2))
      do m = 1, en_no_layers_of_elements(element) - 2
          hiddensizes(m) = en_no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(en_no_layers_of_elements(element)))
      allocate(ohat(en_no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_parameters(element)%en_weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(n, m)
              end do
              if (en_activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (en_activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (en_activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          deallocate(net)
      end do

      force_atomicchi = unraveled_parameters(element)%en_slope * &
      o(layer)%onedarray(1) + unraveled_parameters(element)%en_intercept

      allocate(charge_hiddensizes(charge_no_layers_of_elements(element)&
               - 2))
      do m = 1, charge_no_layers_of_elements(element) - 2
          charge_hiddensizes(m) = charge_no_nodes_of_elements(&
              p_charge + m + 1)
      end do
      allocate(charge_o(charge_no_layers_of_elements(element)))
      allocate(charge_ohat(charge_no_layers_of_elements(element)))
      layer = 1
      allocate(charge_o(1)%onedarray(len_of_charge_fingerprint))
      allocate(charge_ohat(1)%onedarray(len_of_charge_fingerprint + 1))
      do m = 1, (len_of_charge_fingerprint)
          charge_o(1)%onedarray(m) = charge_fingerprint_(m)
      end do
      do layer = 1, size(charge_hiddensizes) + 1
          do m = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim=1) - 1
              charge_ohat(layer)%onedarray(m) = charge_o(&
                  layer)%onedarray(m)
          end do
          charge_ohat(layer)%onedarray(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(charge_net(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(charge_o(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(element)%weights(layer)%twodarray, dim=2)))
          allocate(charge_ohat(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(element)%weights(&
              layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim=2)
              charge_net(m) = 0.0d0
              do n = 1, size(unraveled_charge_parameters(element)%weights(&
              layer)%twodarray, dim=1)
                  charge_net(m) =  charge_net(m) + &
                  charge_ohat(layer)%onedarray(n) * &
                  unraveled_charge_parameters(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (charge_activation_signal == 1) then
                  charge_o(layer + 1)%onedarray(m) = tanh(charge_net(m))
              else if (charge_activation_signal == 2) then
                  charge_o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- charge_net(m)))
              else if (charge_activation_signal == 3) then
                  charge_o(layer + 1)%onedarray(m) = charge_net(m)
              end if
              charge_ohat(layer + 1)%onedarray(m) = charge_o(layer + 1)%onedarray(m)
          end do
          deallocate(charge_net)
      end do

      force_atomiccharge = unraveled_charge_parameters(element)%slope * &
      charge_o(layer)%onedarray(1) + &
      unraveled_charge_parameters(element)%intercept

      nn = size(o) - 2
      allocate(doutputs_dinputs(nn + 2))
      allocate(doutputs_dinputs(1)%onedarray(&
      len_of_fingerprint))
      do m = 1, len_of_fingerprint
      doutputs_dinputs(1)%onedarray(m) = fingerprintprime_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_parameters(element)%en_weights(&
              layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (en_activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (en_activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (en_activation_signal == 3) then
                doutputs_dinputs(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      calculate_force_chi= &
          unraveled_parameters(element)%en_slope * &
          doutputs_dinputs(nn + 2)%onedarray(1)

      calculate_force_chi = -1.0d0 * &
           calculate_force_chi* force_atomiccharge


      nn = size(charge_o) - 2
      allocate(charge_doutputs_dinputs(nn + 2))
      allocate(charge_doutputs_dinputs(1)%onedarray(&
      len_of_charge_fingerprint))
      do m = 1, (len_of_charge_fingerprint)
      charge_doutputs_dinputs(1)%onedarray(m) = charge_fingerprintprime_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim = 2)))
          do p_charge = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim = 2)
              temp(p_charge) = 0.0d0
              do q = 1, size(unraveled_charge_parameters(element)%weights(&
              layer)%twodarray, dim = 1) - 1
                  temp(p_charge) = temp(p_charge) + charge_doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_charge_parameters(&
                  element)%weights(layer)%twodarray(q, p_charge)
              end do
          end do
          q = size(charge_o(layer + 1)%onedarray)
          allocate(charge_doutputs_dinputs(layer + 1)%onedarray(q))
          do p_charge = 1, size(charge_o(layer + 1)%onedarray)
              if (charge_activation_signal == 1) then
                charge_doutputs_dinputs(layer + 1)%onedarray(p_charge) = &
                temp(p_charge) * (1.0d0 - charge_o(layer + 1)%onedarray(&
                p_charge) * charge_o(layer + 1)%onedarray(p_charge))
              else if (charge_activation_signal == 2) then
                charge_doutputs_dinputs(layer + 1)%onedarray(p_charge) = &
                temp(p_charge) * (1.0d0 - charge_o(layer + 1)%onedarray(&
                p_charge)) * charge_o(layer + 1)%onedarray(p_charge)
              else if (charge_activation_signal == 3) then
                charge_doutputs_dinputs(layer+ 1)%onedarray(p_charge) = &
                temp(p_charge)
              end if
          end do
          deallocate(temp)
      end do

      calculate_force_qi= &
          unraveled_charge_parameters(element)%slope * &
          charge_doutputs_dinputs(nn + 2)%onedarray(1)
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      calculate_force_qi = -1.0d0 * &
           calculate_force_qi* (force_atomiccharge *&
           unraveled_parameters(element)%jii + force_atomicchi + &
           work_function)
       
      calculate_atomic_gc_force = calculate_force_qi&
          + calculate_force_chi


!     deallocating neural network
      deallocate(charge_hiddensizes)
      do p = 1, size(charge_o)
          deallocate(charge_o(p)%onedarray)
      end do
      deallocate(charge_o)
      do p = 1, size(charge_ohat)
          deallocate(charge_ohat(p)%onedarray)
      end do
      deallocate(charge_ohat)
      do p = 1, size(charge_doutputs_dinputs)
          deallocate(charge_doutputs_dinputs(p)%onedarray)
      end do
      deallocate(charge_doutputs_dinputs)

!     deallocating derived type parameters
      do element = 1, num_elements
        deallocate(unraveled_charge_parameters(element)%weights)
      end do

      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(doutputs_dinputs)
          deallocate(doutputs_dinputs(p)%onedarray)
      end do
      deallocate(doutputs_dinputs)

!     deallocating derived type parameters
      do element = 1, num_elements
        deallocate(unraveled_parameters(element)%en_weights)
      end do

      end function calculate_atomic_gc_force


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of energy w.r.t. parameters in the
!     atom-centered mode.
      function calculate_dgc_ae_dparameters(symbol, &
      len_of_fingerprint, fingerprint, &
      len_of_charge_fingerprint, charge_fingerprint, &
      work_function, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: num_parameters, num_elements
      integer:: symbol, len_of_fingerprint
      integer:: len_of_charge_fingerprint
      double precision:: &
      calculate_dgc_ae_dparameters(&
      num_parameters+num_parameters)
      double precision:: parameters(num_parameters)
      double precision:: fingerprint(len_of_fingerprint)
      double precision:: charge_fingerprint(len_of_charge_fingerprint)
      double precision:: work_function
      integer:: elements_numbers(num_elements)

      integer:: element, m, n, j, k, l, layer, p, q, nn, num_cols, &
      p_charge, charge_nn
      integer:: num_rows
      double precision:: temp1, temp2
      double precision:: chi, qi
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      integer, allocatable:: charge_hiddensizes(:)
      double precision, allocatable:: charge_net(:)
      type(real_one_d_array), allocatable:: charge_o(:), charge_ohat(:)
      type(real_one_d_array), allocatable:: charge_delta(:), charge_D(:)
      type(element_parameters_four):: &
      unraveled_parameters(num_elements)
      type(element_parameters)::&
      unraveled_charge_parameters(num_elements)
      type(element_parameters_six):: &
      unraveled_daelecneg_dparas(num_elements)
      type(element_parameters_six):: &
      unraveled_dacharge_dparas(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)
      double precision:: charge_fingerprint_(len_of_charge_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do

      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then 
            charge_fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (charge_fingerprint(l) - &
            charge_min_fingerprints(element,l)) / &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprint_(l) = charge_fingerprint(l)
        endif
      end do

      do l = 1, len_of_fingerprint
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - en_min_fingerprints(element, l)) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

!     changing the form of parameters to derived types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(element)%en_weights(&
            j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_parameters(element)%en_weights(j)%twodarray(&
                    p, q) = parameters(l + (p - 1) * num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_parameters(element)%en_intercept = &
        parameters(l + 2 *  element - 1)
        unraveled_parameters(element)%en_slope = &
        parameters(l + 2 * element)
      end do
      l = l + 2 * num_elements

      k = 0
      do element = 1, num_elements
        allocate(unraveled_charge_parameters(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_charge_parameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_charge_parameters(element)%weights(&
                    j)%twodarray(p, q)  = parameters(l + (p - 1) * &
                    num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_charge_parameters(element)%intercept = &
        parameters(l + 2 *  element - 1)
        unraveled_charge_parameters(element)%slope = &
        parameters(l + 2 * element)
      end do
      l = l + 2 * num_elements

      do element = 1, num_elements
        unraveled_parameters(element)%ei = &
        parameters(l + element)
        unraveled_parameters(element)%jii = &
        parameters(l + num_elements + element)
      end do


      do element = 1, num_elements
        unraveled_daelecneg_dparas(element)%intercept = 0.0d0
        unraveled_daelecneg_dparas(element)%slope = 0.0d0
        unraveled_daelecneg_dparas(element)%en_intercept = 0.0d0
        unraveled_daelecneg_dparas(element)%en_slope = 0.0d0
        unraveled_daelecneg_dparas(element)%ei = 0.0d0
        unraveled_daelecneg_dparas(element)%jii = 0.0d0

        unraveled_dacharge_dparas(element)%intercept = 0.0d0
        unraveled_dacharge_dparas(element)%slope = 0.0d0
        unraveled_dacharge_dparas(element)%en_intercept = 0.0d0
        unraveled_dacharge_dparas(element)%en_slope = 0.0d0
        unraveled_dacharge_dparas(element)%ei = 0.0d0
        unraveled_dacharge_dparas(element)%jii = 0.0d0

      end do
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_daelecneg_dparas(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        allocate(unraveled_dacharge_dparas(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_daelecneg_dparas(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            allocate(unraveled_dacharge_dparas(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_daelecneg_dparas(&
                    element)%en_weights(j)%twodarray(p, q) = 0.0d0
                    unraveled_dacharge_dparas(&
                    element)%en_weights(j)%twodarray(p, q) = 0.0d0
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      l = l + 2 * num_elements
      k = 0
      do element = 1, num_elements
        allocate(unraveled_daelecneg_dparas(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        allocate(unraveled_dacharge_dparas(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_daelecneg_dparas(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            allocate(unraveled_dacharge_dparas(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_daelecneg_dparas(element)%weights(&
                    j)%twodarray(p, q)  = 0.0d0
                    unraveled_dacharge_dparas(element)%weights(&
                    j)%twodarray(p, q)  = 0.0d0
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do

      p = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p = p + en_no_layers_of_elements(element)
          end if
      end do

      p_charge = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p_charge = p_charge + &
                charge_no_layers_of_elements(element)
          end if
      end do

      allocate(hiddensizes(en_no_layers_of_elements(element) - 2))
      do m = 1, en_no_layers_of_elements(element) - 2
          hiddensizes(m) = en_no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(en_no_layers_of_elements(element)))
      allocate(ohat(en_no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_parameters(element)%en_weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(n, m)
              end do
              if (en_activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (en_activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (en_activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do
      chi = unraveled_parameters(element)%en_slope * &
        o(layer)%onedarray(1) + &
        unraveled_parameters(element)%en_intercept

      allocate(charge_hiddensizes(charge_no_layers_of_elements(element)&
               - 2))
      do m = 1, charge_no_layers_of_elements(element) - 2
          charge_hiddensizes(m) = charge_no_nodes_of_elements(&
              p_charge + m + 1)
      end do
      allocate(charge_o(charge_no_layers_of_elements(element)))
      allocate(charge_ohat(charge_no_layers_of_elements(element)))
      layer = 1
      allocate(charge_o(1)%onedarray(len_of_charge_fingerprint))
      allocate(charge_ohat(1)%onedarray(len_of_charge_fingerprint + 1))
      do m = 1, (len_of_charge_fingerprint)
          charge_o(1)%onedarray(m) = charge_fingerprint_(m)
      end do

      do layer = 1, size(charge_hiddensizes) + 1
          do m = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim=1) - 1
              charge_ohat(layer)%onedarray(m) = charge_o(&
                  layer)%onedarray(m)
          end do
          charge_ohat(layer)%onedarray(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(charge_net(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(charge_o(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(element)%weights(layer)%twodarray,dim=2)))
          allocate(charge_ohat(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(element)%weights(&
              layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim=2)
              charge_net(m) = 0.0d0
              do n = 1, size(unraveled_charge_parameters(element)%weights(&
              layer)%twodarray, dim=1)
                  charge_net(m) =  charge_net(m) + &
                  charge_ohat(layer)%onedarray(n) * &
                  unraveled_charge_parameters(&
                  element)%weights(layer)%twodarray(n, m)
              end do
              if (charge_activation_signal == 1) then
                  charge_o(layer + 1)%onedarray(m) = tanh(charge_net(m))
              else if (charge_activation_signal == 2) then
                  charge_o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- charge_net(m)))
              else if (charge_activation_signal == 3) then
                  charge_o(layer + 1)%onedarray(m) = charge_net(m)
              end if
              charge_ohat(layer + 1)%onedarray(m) = &
                charge_o(layer +1)%onedarray(m)
          end do
          deallocate(charge_net)
      end do
      qi = unraveled_charge_parameters(element)%slope * &
        charge_o(layer)%onedarray(1) + &
        unraveled_charge_parameters(element)%intercept

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (en_activation_signal == 1) then
                  D(layer)%onedarray(j) = (1.0d0 - &
                  o(layer + 1)%onedarray(j)* o(layer + 1)%onedarray(j))
              elseif (en_activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (en_activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * unraveled_parameters(&
                  element)%en_weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      charge_nn = size(charge_o) - 2
      allocate(charge_D(charge_nn + 1))
      do layer = 1, charge_nn + 1
          allocate(charge_D(layer)%onedarray(size(&
            charge_o(layer + 1)%onedarray)))
          do j = 1, size(charge_o(layer + 1)%onedarray)
              if (charge_activation_signal == 1) then
                  charge_D(layer)%onedarray(j) = (1.0d0 - &
                  charge_o(layer + 1)%onedarray(j)* &
                  charge_o(layer + 1)%onedarray(j))
              elseif (charge_activation_signal == 2) then
                  charge_D(layer)%onedarray(j) = &
                  charge_o(layer + 1)%onedarray(j) * &
                  (1.0d0 - charge_o(layer + 1)%onedarray(j))
              elseif (charge_activation_signal == 3) then
                  charge_D(layer)%onedarray(j) = 1.0d0
              end if
          end do
      end do
      allocate(charge_delta(charge_nn + 1))
      allocate(charge_delta(charge_nn + 1)%onedarray(1))
      charge_delta(charge_nn + 1)%onedarray(1) = &
        charge_D(charge_nn + 1)%onedarray(1)
      do layer = charge_nn, 1, -1
          allocate(charge_delta(layer)%onedarray(&
            size(charge_D(layer)%onedarray)))
          do p = 1, size(charge_D(layer)%onedarray)
              charge_delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(charge_delta(layer + 1)%onedarray)
                  temp1 = charge_D(layer)%onedarray(p) * &
                  unraveled_charge_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * charge_delta(layer + 1)%onedarray(q)
                  charge_delta(layer)%onedarray(p) = &
                  charge_delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      unraveled_daelecneg_dparas(element)%intercept = chi + &
        unraveled_parameters(element)%jii * qi + work_function 
      unraveled_daelecneg_dparas(element)%slope = &
        charge_o(charge_nn + 2)%onedarray(1) * (chi + &
        unraveled_parameters(element)%jii * qi + &
        work_function)
      unraveled_daelecneg_dparas(element)%en_intercept = qi 
      unraveled_daelecneg_dparas(element)%en_slope = &
        o(nn + 2)%onedarray(1) * qi
      unraveled_daelecneg_dparas(element)%jii = 0.5d0 * (qi ** 2.0d0)
      unraveled_daelecneg_dparas(element)%ei = 1.0d0

      unraveled_dacharge_dparas(element)%intercept = 1.0d0
      unraveled_dacharge_dparas(element)%slope = &
        charge_o(charge_nn + 2)%onedarray(1) 


      do layer = 1, nn + 1
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  unraveled_daelecneg_dparas(element)%en_weights(&
                  layer)%twodarray(p, q) = qi * &
                  unraveled_parameters(element)%en_slope * &
                  ohat(layer)%onedarray(p) * delta(layer)%onedarray(q)
              end do
          end do
      end do

      do layer = 1, charge_nn + 1
          do p = 1, size(charge_ohat(layer)%onedarray)
              do q = 1, size(charge_delta(layer)%onedarray)
                  unraveled_daelecneg_dparas(element)%weights(&
                  layer)%twodarray(p, q) = (chi + &
                  unraveled_parameters(element)%jii * qi + &
                  work_function) * &
                  unraveled_charge_parameters(element)%slope * &
                  charge_ohat(layer)%onedarray(p) * &
                  charge_delta(layer)%onedarray(q)

                  unraveled_dacharge_dparas(element)%weights(&
                  layer)%twodarray(p, q) = &
                  unraveled_charge_parameters(element)%slope * &
                  charge_ohat(layer)%onedarray(p) * &
                  charge_delta(layer)%onedarray(q)
              end do
          end do
      end do

      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)

      deallocate(charge_hiddensizes)
      do p = 1, size(charge_o)
          deallocate(charge_o(p)%onedarray)
      end do
      deallocate(charge_o)
      do p = 1, size(charge_ohat)
          deallocate(charge_ohat(p)%onedarray)
      end do
      deallocate(charge_ohat)
      do p = 1, size(charge_delta)
          deallocate(charge_delta(p)%onedarray)
      end do
      deallocate(charge_delta)
      do p = 1, size(charge_D)
          deallocate(charge_D(p)%onedarray)
      end do
      deallocate(charge_D)

!     changing the derivatives of the energy from derived-type
!     form into vector
      k = 0
      l = 0
      do element = 1, num_elements
        if (element > 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_dgc_ae_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_daelecneg_dparas(&
                    element)%en_weights(j)%twodarray(p, q)

                    calculate_dgc_ae_dparameters(&
                    l + (p - 1) * num_cols + q + &
                    num_parameters) = &
                    unraveled_dacharge_dparas(&
                    element)%en_weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_dgc_ae_dparameters(l + 2 *  element - 1) = &
        unraveled_daelecneg_dparas(element)%en_intercept
        calculate_dgc_ae_dparameters(l + 2 * element) = &
        unraveled_daelecneg_dparas(element)%en_slope

        calculate_dgc_ae_dparameters(l + 2 *  element - 1 + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%en_intercept
        calculate_dgc_ae_dparameters(l + 2 * element + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%en_slope
      end do
      l = l + 2 * num_elements

      k=0
      do element = 1, num_elements
        if (element > 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_dgc_ae_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_daelecneg_dparas(&
                    element)%weights(j)%twodarray(p, q)

                    calculate_dgc_ae_dparameters(&
                    l + (p - 1) * num_cols + q + &
                    num_parameters) = &
                    unraveled_dacharge_dparas(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_dgc_ae_dparameters(l + 2 *  element - 1) = &
        unraveled_daelecneg_dparas(element)%intercept
        calculate_dgc_ae_dparameters(l + 2 * element) = &
        unraveled_daelecneg_dparas(element)%slope

        calculate_dgc_ae_dparameters(l + 2 *  element - 1 + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%intercept
        calculate_dgc_ae_dparameters(l + 2 * element + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%slope
      end do
      l = l + 2 * num_elements

      do element = 1, num_elements
        calculate_dgc_ae_dparameters(l + element) = &
        unraveled_daelecneg_dparas(element)%ei
        calculate_dgc_ae_dparameters(l + element + num_elements) = &
        unraveled_daelecneg_dparas(element)%jii

        calculate_dgc_ae_dparameters(l + element + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%ei
        calculate_dgc_ae_dparameters(l + element + num_elements + &
        num_parameters) = &
        unraveled_dacharge_dparas(element)%jii
      end do

!     deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
          num_rows = en_no_nodes_of_elements(k + j) + 1
          num_cols = en_no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%en_weights(j)%twodarray)
          deallocate(unraveled_daelecneg_dparas(&
          element)%en_weights(j)%twodarray)
          deallocate(unraveled_dacharge_dparas(&
          element)%en_weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%en_weights)
        deallocate(unraveled_daelecneg_dparas(element)%en_weights)
        deallocate(unraveled_dacharge_dparas(element)%en_weights)
      end do

      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
          num_rows = charge_no_nodes_of_elements(k + j) + 1
          num_cols = charge_no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_charge_parameters(&
          element)%weights(j)%twodarray)
          deallocate(unraveled_daelecneg_dparas(&
          element)%weights(j)%twodarray)
          deallocate(unraveled_dacharge_dparas(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_charge_parameters(element)%weights)
        deallocate(unraveled_daelecneg_dparas(element)%weights)
        deallocate(unraveled_dacharge_dparas(element)%weights)
      end do

      end function calculate_dgc_ae_dparameters


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of force w.r.t. parameters in the
!     atom-centered mode
      function calculate_dgcforce_dparameters(symbol, &
      len_of_fingerprint, &
      fingerprint, fingerprintprime, &
      len_of_charge_fingerprint, charge_fingerprint, &
      charge_fingerprintprime, &
      work_function, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, len_of_fingerprint, &
      len_of_charge_fingerprint, num_parameters
      integer:: num_elements
      double precision:: fingerprint(len_of_fingerprint)
      double precision:: fingerprintprime(len_of_fingerprint)
      double precision:: charge_fingerprint(len_of_charge_fingerprint)
      double precision:: charge_fingerprintprime(&
                         len_of_charge_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_dgcforce_dparameters(num_parameters)
      double precision:: qi, chi, work_function
      double precision:: dq_slope_dinputs, q_slope, tmp1, tmp2

      integer:: element, m, n, j, k, l, layer, p, q, nn, num_cols
      integer:: num_rows, p_charge, charge_nn
      double precision:: temp1, temp2
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(real_one_d_array), allocatable:: doutputs_dinputs(:)
      double precision, allocatable:: dohat_dinputs(:)
      type(real_one_d_array), allocatable:: dD_dinputs(:)
      type(real_one_d_array), allocatable:: ddelta_dinputs(:)
      double precision, allocatable:: &
      doutput_dinputsdweights(:, :)
      integer, allocatable:: charge_hiddensizes(:)
      double precision, allocatable:: charge_net(:)
      type(real_one_d_array), allocatable:: charge_o(:), charge_ohat(:)
      type(real_one_d_array), allocatable:: charge_delta(:), charge_D(:)
      type(real_one_d_array), allocatable:: charge_doutputs_dinputs(:)
      double precision, allocatable:: charge_dohat_dinputs(:)
      type(real_one_d_array), allocatable:: charge_dD_dinputs(:)
      type(real_one_d_array), allocatable:: charge_ddelta_dinputs(:)
      double precision, allocatable:: &
      charge_doutput_dinputsdweights(:, :)
      double precision, allocatable:: temp(:), temp3(:), temp4(:)
      double precision, allocatable:: temp5(:), temp6(:)
      type(element_parameters_four):: unraveled_parameters(num_elements)
      type(element_parameters):: unraveled_charge_parameters(num_elements)  
      type(element_parameters_six):: &
      unraveled_denforce_dparameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)
      double precision:: fingerprintprime_(len_of_fingerprint)
      double precision:: charge_fingerprint_(len_of_charge_fingerprint)
      double precision:: &
      charge_fingerprintprime_(len_of_charge_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do

      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            charge_fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (charge_fingerprint(l)-charge_min_fingerprints(element,l))/ &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprint_(l) = charge_fingerprint(l)
        endif
      end do

      do l = 1, (len_of_fingerprint)
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - en_min_fingerprints(element,l)) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

      ! scaling fingerprintprimes
      do l = 1, len_of_fingerprint
        if ((en_max_fingerprints(element, l) - &
        en_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprintprime_(l) = &
            2.0d0 * fingerprintprime(l) / &
            (en_max_fingerprints(element, l) - &
            en_min_fingerprints(element, l))
        else
            fingerprintprime_(l) = fingerprintprime(l)
        endif
      end do

      do l = 1, (len_of_charge_fingerprint)
        if ((charge_max_fingerprints(element, l) - &
        charge_min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            charge_fingerprintprime_(l) = &
            2.0d0 * charge_fingerprintprime(l) / &
            (charge_max_fingerprints(element, l) - &
            charge_min_fingerprints(element, l))
        else
            charge_fingerprintprime_(l) = charge_fingerprintprime(l)
        endif
      end do


!     changing the form of parameters from vector into derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_parameters(element)%en_weights(j)%twodarray(&
                    p, q) = parameters(l + (p - 1) * num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do

      do element = 1, num_elements
        unraveled_parameters(element)%en_intercept = &
        parameters(l + 2 * element - 1)
        unraveled_parameters(element)%en_slope = &
        parameters(l + 2 * element )
      end do
      l = l + 2 * num_elements

      k = 0
      do element = 1, num_elements
        allocate(unraveled_charge_parameters(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_charge_parameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_charge_parameters(element)%weights(&
                    j)%twodarray(p, q)  = parameters(l + (p - 1) * &
                    num_cols + q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        unraveled_charge_parameters(element)%intercept = &
        parameters(l + 2 *  element - 1)
        unraveled_charge_parameters(element)%slope = &
        parameters(l + 2 * element)
      end do
      l = l + 2 * num_elements


      do element = 1, num_elements
        unraveled_parameters(element)%ei = &
        parameters(l + element)
        unraveled_parameters(element)%jii = &
        parameters(l + num_elements + element)
      end do


      do element = 1, num_elements
        unraveled_denforce_dparameters(element)%intercept = 0.0d0
        unraveled_denforce_dparameters(element)%slope = 0.0d0
        unraveled_denforce_dparameters(element)%en_intercept = 0.0d0
        unraveled_denforce_dparameters(element)%en_slope = 0.0d0
        unraveled_denforce_dparameters(element)%ei = 0.0d0
        unraveled_denforce_dparameters(element)%jii = 0.0d0
      end do

      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_denforce_dparameters(element)%en_weights(&
        en_no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_denforce_dparameters(&
            element)%en_weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_denforce_dparameters(&
                    element)%en_weights(j)%twodarray(p, q) = 0.0d0
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      l = l + 2 * num_elements
      k = 0

      do element = 1, num_elements
        allocate(unraveled_denforce_dparameters(element)%weights(&
        charge_no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            allocate(unraveled_denforce_dparameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_denforce_dparameters(&
                    element)%weights(j)%twodarray(p, q) = 0.0d0
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do

      p = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p = p + en_no_layers_of_elements(element)
          end if
      end do

      p_charge = 0
      do element = 1, num_elements
          if (symbol == elements_numbers(element)) then
              exit
          else
              p_charge = p_charge + charge_no_layers_of_elements(&
                         element)
          end if
      end do

      allocate(hiddensizes(en_no_layers_of_elements(element) - 2))
      do m = 1, en_no_layers_of_elements(element) - 2
          hiddensizes(m) = en_no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(en_no_layers_of_elements(element)))
      allocate(ohat(en_no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_parameters(element)%en_weights(&
          layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(unraveled_parameters(element)%en_weights(&
              layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(n, m)
              end do
              if (en_activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (en_activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (en_activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      chi = unraveled_parameters(element)%en_slope * &
          o(layer)%onedarray(1) + &
          unraveled_parameters(element)%en_intercept

      allocate(charge_hiddensizes(charge_no_layers_of_elements(element)&
        - 2))
      do m = 1, charge_no_layers_of_elements(element) - 2
          charge_hiddensizes(m) = charge_no_nodes_of_elements(p_charge+&
           m + 1)
      end do
      allocate(charge_o(charge_no_layers_of_elements(element)))
      allocate(charge_ohat(charge_no_layers_of_elements(element)))
      layer = 1
      allocate(charge_o(1)%onedarray(len_of_charge_fingerprint))
      allocate(charge_ohat(1)%onedarray(len_of_charge_fingerprint + 1))
      do m = 1, len_of_charge_fingerprint
          charge_o(1)%onedarray(m) = charge_fingerprint_(m)
      end do
      do layer = 1, size(charge_hiddensizes) + 1
          do m = 1, size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=1) - 1
              charge_ohat(layer)%onedarray(m) = &
              charge_o(layer)%onedarray(m)
          end do
          charge_ohat(layer)%onedarray(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(charge_net(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(charge_o(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2)))
          allocate(charge_ohat(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(unraveled_charge_parameters(element)%weights(&
          layer)%twodarray, dim=2)
              charge_net(m) = 0.0d0
              do n = 1, size(unraveled_charge_parameters(&
              element)%weights(layer)%twodarray, dim=1)
                  charge_net(m) =  charge_net(m) + &
                  charge_ohat(layer)%onedarray(n) * &
                  unraveled_charge_parameters(element)%weights(&
                  layer)%twodarray(n, m)
              end do
              if (charge_activation_signal == 1) then
                  charge_o(layer + 1)%onedarray(m) = tanh(&
                  charge_net(m))
              else if (charge_activation_signal == 2) then
                  charge_o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- charge_net(m)))
              else if (charge_activation_signal == 3) then
                  charge_o(layer + 1)%onedarray(m) = charge_net(m)
              end if
              charge_ohat(layer + 1)%onedarray(m) = &
              charge_o(layer + 1)%onedarray(m)
          end do
          charge_ohat(layer + 1)%onedarray(size(&
          unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(charge_net)
      end do

      qi = unraveled_charge_parameters(element)%slope * &
         charge_o(layer)%onedarray(1) + &
         unraveled_charge_parameters(element)%intercept

      nn = size(o) - 2

      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (en_activation_signal == 1) then
                  D(layer)%onedarray(j) = &
                  (1.0d0 - o(layer + 1)%onedarray(j)* &
                  o(layer + 1)%onedarray(j))
              elseif (en_activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (en_activation_signal == 3) then
                  D(layer)%onedarray(j) = 1.0d0
              end if
          end do
      end do
      allocate(delta(nn + 1))
      allocate(delta(nn + 1)%onedarray(1))
      delta(nn + 1)%onedarray(1) = D(nn + 1)%onedarray(1)
      do layer = nn, 1, -1
          allocate(delta(layer)%onedarray(size(D(layer)%onedarray)))
          do p = 1, size(D(layer)%onedarray)
              delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp1 = D(layer)%onedarray(p) * &
                  unraveled_parameters(element)%en_weights(&
                  layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      allocate(doutputs_dinputs(nn + 2))
      allocate(doutputs_dinputs(1)%onedarray(&
      len_of_fingerprint))
      do m = 1, len_of_fingerprint
        doutputs_dinputs(1)%onedarray(m) = fingerprintprime_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_parameters(&
          element)%en_weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_parameters(&
              element)%en_weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_parameters(&
                  element)%en_weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (en_activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (en_activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (en_activation_signal == 3) then
                doutputs_dinputs(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do
      allocate(dD_dinputs(nn + 1))
      do layer = 1, nn + 1
          allocate(dD_dinputs(layer)%onedarray(&
          size(o(layer + 1)%onedarray)))
          do p = 1, size(o(layer + 1)%onedarray)
              if (en_activation_signal == 1) then
                  dD_dinputs(layer)%onedarray(p) =- 2.0d0 * &
                  o(layer + 1)%onedarray(p) * &
                  doutputs_dinputs(layer + 1)%onedarray(p)
              elseif (en_activation_signal == 2) then
                  dD_dinputs(layer)%onedarray(p) = &
                  doutputs_dinputs(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * o(layer + 1)%onedarray(p))
              elseif (en_activation_signal == 3) then
                  dD_dinputs(layer)%onedarray(p) =0.0d0
              end if
          end do
      end do


      allocate(ddelta_dinputs(nn + 1))
      allocate(ddelta_dinputs(nn + 1)%onedarray(1))
      ddelta_dinputs(nn + 1)%onedarray(1) = &
      dD_dinputs(nn + 1)%onedarray(1)

      do layer = nn, 1, -1
          allocate(temp3(size(unraveled_parameters(element)%en_weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(size(unraveled_parameters(element)%en_weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(unraveled_parameters(element)%en_weights(&
          layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + unraveled_parameters(&
                  element)%en_weights(layer + 1)%twodarray(p, q) * &
                  delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + unraveled_parameters(&
                  element)%en_weights(layer + 1)%twodarray(p, q) * &
                  ddelta_dinputs(layer + 1)%onedarray(q)
              end do
          end do
          allocate(temp5(size(dD_dinputs(layer)%onedarray)))
          allocate(temp6(size(dD_dinputs(layer)%onedarray)))
          allocate(ddelta_dinputs(layer)%onedarray(&
          size(dD_dinputs(layer)%onedarray)))
          do p = 1, size(dD_dinputs(layer)%onedarray)
              temp5(p) = &
              dD_dinputs(layer)%onedarray(p) * temp3(p)
              temp6(p) = D(layer)%onedarray(p) * temp4(p)
              ddelta_dinputs(layer)%onedarray(p)= &
              temp5(p) + temp6(p)
          end do
          deallocate(temp3)
          deallocate(temp4)
          deallocate(temp5)
          deallocate(temp6)
      end do


      charge_nn = size(charge_o) - 2
      allocate(charge_D(charge_nn + 1))
      do layer = 1, charge_nn + 1
          allocate(charge_D(layer)%onedarray(size(&
            charge_o(layer + 1)%onedarray)))
          do j = 1, size(charge_o(layer + 1)%onedarray)
              if (charge_activation_signal == 1) then
                  charge_D(layer)%onedarray(j) = (1.0d0 - &
                  charge_o(layer + 1)%onedarray(j)* &
                  charge_o(layer + 1)%onedarray(j))
              elseif (charge_activation_signal == 2) then
                  charge_D(layer)%onedarray(j) = &
                  charge_o(layer + 1)%onedarray(j) * &
                  (1.0d0 - charge_o(layer + 1)%onedarray(j))
              elseif (charge_activation_signal == 3) then
                  charge_D(layer)%onedarray(j) = 1.0d0
              end if
          end do
      end do
      allocate(charge_delta(charge_nn + 1))
      allocate(charge_delta(charge_nn + 1)%onedarray(1))
      charge_delta(charge_nn + 1)%onedarray(1) = &
        charge_D(charge_nn + 1)%onedarray(1)
      do layer = charge_nn, 1, -1
          allocate(charge_delta(layer)%onedarray(&
            size(charge_D(layer)%onedarray)))
          do p = 1, size(charge_D(layer)%onedarray)
              charge_delta(layer)%onedarray(p) = 0.0d0
              do q = 1, size(charge_delta(layer + 1)%onedarray)
                  temp1 = charge_D(layer)%onedarray(p) * &
                  unraveled_charge_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * charge_delta(layer + 1)%onedarray(q)
                  charge_delta(layer)%onedarray(p) = &
                  charge_delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do


      allocate(charge_doutputs_dinputs(charge_nn + 2))
      allocate(charge_doutputs_dinputs(1)%onedarray(&
      len_of_charge_fingerprint))
      do m = 1, len_of_charge_fingerprint
        charge_doutputs_dinputs(1)%onedarray(m) = &
        charge_fingerprintprime_(m)
      end do
      do layer = 1, charge_nn + 1
          allocate(temp(size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_charge_parameters(&
          element)%weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_charge_parameters(&
              element)%weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + charge_doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_charge_parameters(&
                  element)%weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(charge_o(layer + 1)%onedarray)
          allocate(charge_doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(charge_o(layer + 1)%onedarray)
              if (charge_activation_signal == 1) then
                charge_doutputs_dinputs(&
                layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - charge_o(layer + 1)%onedarray(p) * &
                charge_o(layer + 1)%onedarray(p))
              else if (charge_activation_signal == 2) then
                charge_doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * &
                (1.0d0 - charge_o(layer + 1)%onedarray(p)) * &
                charge_o(layer + 1)%onedarray(p)
              else if (charge_activation_signal == 3) then
                charge_doutputs_dinputs(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      allocate(charge_dD_dinputs(charge_nn + 1))
      do layer = 1, charge_nn + 1
          allocate(charge_dD_dinputs(layer)%onedarray(&
          size(charge_o(layer + 1)%onedarray)))
          do p = 1, size(charge_o(layer + 1)%onedarray)
              if (charge_activation_signal == 1) then
                  charge_dD_dinputs(layer)%onedarray(p) =- 2.0d0 * &
                  charge_o(layer + 1)%onedarray(p) * &
                  charge_doutputs_dinputs(layer + 1)%onedarray(p)
              elseif (charge_activation_signal == 2) then
                  charge_dD_dinputs(layer)%onedarray(p) = &
                  charge_doutputs_dinputs(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * charge_o(layer + 1)%onedarray(p))
              elseif (charge_activation_signal == 3) then
                  charge_dD_dinputs(layer)%onedarray(p) =0.0d0
              end if
          end do
      end do

      allocate(charge_ddelta_dinputs(charge_nn + 1))
      allocate(charge_ddelta_dinputs(charge_nn + 1)%onedarray(1))
      charge_ddelta_dinputs(charge_nn + 1)%onedarray(1) = &
      charge_dD_dinputs(charge_nn + 1)%onedarray(1)

      do layer = charge_nn, 1, -1
          allocate(temp3(size(unraveled_charge_parameters(&
          element)%weights(layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(size(unraveled_charge_parameters(&
          element)%weights(layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(unraveled_charge_parameters(&
          element)%weights(layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(charge_delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + unraveled_charge_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
                  charge_delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + unraveled_charge_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
                  charge_ddelta_dinputs(layer + 1)%onedarray(q)
              end do
          end do
          allocate(temp5(size(charge_dD_dinputs(layer)%onedarray)))
          allocate(temp6(size(charge_dD_dinputs(layer)%onedarray)))
          allocate(charge_ddelta_dinputs(layer)%onedarray(&
          size(charge_dD_dinputs(layer)%onedarray)))
          do p = 1, size(charge_dD_dinputs(layer)%onedarray)
              temp5(p) = &
              charge_dD_dinputs(layer)%onedarray(p) * temp3(p)
              temp6(p) = charge_D(layer)%onedarray(p) * temp4(p)
              charge_ddelta_dinputs(layer)%onedarray(p)= &
              temp5(p) + temp6(p)
          end do
          deallocate(temp3)
          deallocate(temp4)
          deallocate(temp5)
          deallocate(temp6)
      end do


      q_slope = chi + unraveled_parameters(element)%jii * qi + &
                work_function
      

      dq_slope_dinputs = unraveled_parameters(element)%en_slope * &
                         doutputs_dinputs(nn + 2)%onedarray(1) + &
                         unraveled_parameters(element)%jii * &
                         unraveled_charge_parameters(element)%slope * &
                         charge_doutputs_dinputs(&
                         charge_nn + 2)%onedarray(1)
      
      unraveled_denforce_dparameters(element)%en_slope = &
      doutputs_dinputs(nn + 2) %onedarray(1) * qi + &
      unraveled_charge_parameters(element)%slope * &
      o(nn + 2)%onedarray(1) * &
      charge_doutputs_dinputs(charge_nn + 2)%onedarray(1)      
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      unraveled_denforce_dparameters(element)%en_slope = &
      -1.0d0 * unraveled_denforce_dparameters(element)%en_slope 
 
      unraveled_denforce_dparameters(element)%slope = &
      charge_doutputs_dinputs(charge_nn + 2)%onedarray(1) * q_slope + &
      charge_o(size(charge_hiddensizes) + 1)%onedarray(1) * &
      dq_slope_dinputs
      unraveled_denforce_dparameters(element)%slope = & 
      -1.0d0 * unraveled_denforce_dparameters(element)%slope

      unraveled_denforce_dparameters(element)%jii = qi * &
      unraveled_charge_parameters(element)%slope * &
      charge_doutputs_dinputs(charge_nn + 2)%onedarray(1)
      unraveled_denforce_dparameters(element)%jii = &
      -1.0d0 * unraveled_denforce_dparameters(element)%jii

      unraveled_denforce_dparameters(element)%en_intercept = &
      unraveled_charge_parameters(element)%slope * &
      charge_doutputs_dinputs(charge_nn + 2)%onedarray(1)
      unraveled_denforce_dparameters(element)%en_intercept = & 
      -1.0d0 * unraveled_denforce_dparameters(element)%en_intercept


      unraveled_denforce_dparameters(element)%intercept = &
      dq_slope_dinputs
      unraveled_denforce_dparameters(element)%intercept = &
      -1.0d0 * unraveled_denforce_dparameters(element)%intercept

      do layer = 1, nn + 1
          allocate(dohat_dinputs(&
          size(doutputs_dinputs(layer)%onedarray) + 1))
          do p = 1, size(doutputs_dinputs(layer)%onedarray)
              dohat_dinputs(p) = &
              doutputs_dinputs(layer)%onedarray(p)
          end do
          dohat_dinputs(&
          size(doutputs_dinputs(layer)%onedarray) + 1) = 0.0d0
          allocate(doutput_dinputsdweights(&
          size(dohat_dinputs), size(delta(layer)%onedarray)))
          do p = 1, size(dohat_dinputs)
              do q = 1, size(delta(layer)%onedarray)
                  doutput_dinputsdweights(p, q)= 0.0d0
              end do
          end do
          do p = 1, size(dohat_dinputs)
              do q = 1, size(delta(layer)%onedarray)
              doutput_dinputsdweights(p, q) = &
              doutput_dinputsdweights(p, q) + &
              dohat_dinputs(p) * delta(layer)%onedarray(q) + &
              ohat(layer)%onedarray(p)* &
              ddelta_dinputs(layer)%onedarray(q)
              end do
          end do

          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  unraveled_denforce_dparameters(element)%en_weights(&
                  layer)%twodarray(p, q) = &
                  unraveled_parameters(element)%en_slope * &
                  unraveled_charge_parameters(element)%slope * &
                  ohat(layer)%onedarray(p) * &
                  delta(layer)%onedarray(q) * &
                  charge_doutputs_dinputs(charge_nn + 2)%onedarray(&
                  1) + unraveled_parameters(element)%en_slope * &
                  doutput_dinputsdweights(p, q) * qi
          !        ! force is multiplied by -1, because it is -dE/dx and
          !        ! not dE/dx.
                  unraveled_denforce_dparameters(element)%en_weights(&
                  layer)%twodarray(p, q) = &
                  -1.0d0 * unraveled_denforce_dparameters(&
                  element)%en_weights(layer)%twodarray(p, q)
              end do
          end do
          deallocate(dohat_dinputs)
          deallocate(doutput_dinputsdweights)
      end do
      do layer = 1, charge_nn + 1
          allocate(charge_dohat_dinputs(&
          size(charge_doutputs_dinputs(layer)%onedarray) + 1))
          do p = 1, size(charge_doutputs_dinputs(layer)%onedarray)
              charge_dohat_dinputs(p) = &
              charge_doutputs_dinputs(layer)%onedarray(p)
          end do
          charge_dohat_dinputs(&
          size(charge_doutputs_dinputs(layer)%onedarray) + 1) = 0.0d0
          allocate(charge_doutput_dinputsdweights(&
          size(charge_dohat_dinputs), &
          size(charge_delta(layer)%onedarray)))
          do p = 1, size(charge_dohat_dinputs)
              do q = 1, size(charge_delta(layer)%onedarray)
                  charge_doutput_dinputsdweights(p, q)= 0.0d0
              end do
          end do
          do p = 1, size(charge_dohat_dinputs)
              do q = 1, size(charge_delta(layer)%onedarray)
              charge_doutput_dinputsdweights(p, q) = &
              charge_doutput_dinputsdweights(p, q) + &
              charge_dohat_dinputs(p) * &
              charge_delta(layer)%onedarray(q) + &
              charge_ohat(layer)%onedarray(p)* &
              charge_ddelta_dinputs(layer)%onedarray(q)
              end do
          end do
          do p = 1, size(charge_ohat(layer)%onedarray)
              do q = 1, size(charge_delta(layer)%onedarray)
              unraveled_denforce_dparameters(element)%weights(&
              layer)%twodarray(p, q) = q_slope * &
              charge_doutput_dinputsdweights(p, q) * & 
              unraveled_charge_parameters(element)%slope + &
              unraveled_charge_parameters(element)%slope * &
              charge_ohat(layer)%onedarray(p) * &
              charge_delta(layer)%onedarray(q) * &
              dq_slope_dinputs
              ! force is multiplied by -1, because it is -dE/dx and
              ! not dE/dx.
              unraveled_denforce_dparameters(element)%weights(&
              layer)%twodarray(p, q) = &
              -1.0d0 * unraveled_denforce_dparameters(element)%weights(&
              layer)%twodarray(p, q)
              end do
          end do
          deallocate(charge_dohat_dinputs)
          deallocate(charge_doutput_dinputsdweights)
      end do

!     deallocating neural network
      deallocate(hiddensizes)
      do p = 1, size(o)
          deallocate(o(p)%onedarray)
      end do
      deallocate(o)
      do p = 1, size(ohat)
          deallocate(ohat(p)%onedarray)
      end do
      deallocate(ohat)
      do p = 1, size(delta)
          deallocate(delta(p)%onedarray)
      end do
      deallocate(delta)
      do p = 1, size(D)
          deallocate(D(p)%onedarray)
      end do
      deallocate(D)
      do p = 1, size(doutputs_dinputs)
          deallocate(doutputs_dinputs(p)%onedarray)
      end do
      deallocate(doutputs_dinputs)
      do p = 1, size(ddelta_dinputs)
          deallocate(ddelta_dinputs(p)%onedarray)
      end do
      deallocate(ddelta_dinputs)
      do p = 1, size(dD_dinputs)
          deallocate(dD_dinputs(p)%onedarray)
      end do
      deallocate(dD_dinputs)

      deallocate(charge_hiddensizes)
      do p = 1, size(charge_o)
          deallocate(charge_o(p)%onedarray)
      end do
      deallocate(charge_o)
      do p = 1, size(charge_ohat)
          deallocate(charge_ohat(p)%onedarray)
      end do
      deallocate(charge_ohat)
      do p = 1, size(charge_delta)
          deallocate(charge_delta(p)%onedarray)
      end do
      deallocate(charge_delta)
      do p = 1, size(charge_D)
          deallocate(charge_D(p)%onedarray)
      end do
      deallocate(charge_D)
      do p = 1, size(charge_doutputs_dinputs)
          deallocate(charge_doutputs_dinputs(p)%onedarray)
      end do
      deallocate(charge_doutputs_dinputs)
      do p = 1, size(charge_ddelta_dinputs)
          deallocate(charge_ddelta_dinputs(p)%onedarray)
      end do
      deallocate(charge_ddelta_dinputs)
      do p = 1, size(charge_dD_dinputs)
          deallocate(charge_dD_dinputs(p)%onedarray)
      end do
      deallocate(charge_dD_dinputs)
      k = 0
      l = 0
      do element = 1, num_elements
        if (element > 1) then
            k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
            num_rows = en_no_nodes_of_elements(k + j) + 1
            num_cols = en_no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_dgcforce_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_denforce_dparameters(&
                    element)%en_weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_dgcforce_dparameters(l + 2 *  element - 1) = &
        unraveled_denforce_dparameters(element)%en_intercept
        calculate_dgcforce_dparameters(l + 2 * element) = &
        unraveled_denforce_dparameters(element)%en_slope
      end do
      l = l + 2 * num_elements

      k = 0
      do element = 1, num_elements
        if (element > 1) then
            k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
            num_rows = charge_no_nodes_of_elements(k + j) + 1
            num_cols = charge_no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_dgcforce_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_denforce_dparameters(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_dgcforce_dparameters(l + 2 *  element - 1) = &
        unraveled_denforce_dparameters(element)%intercept
        calculate_dgcforce_dparameters(l + 2 * element) = &
        unraveled_denforce_dparameters(element)%slope
      end do
      l = l + 2 * num_elements

      do element = 1, num_elements
        calculate_dgcforce_dparameters(l+ element) = &
          unraveled_denforce_dparameters(element)%ei
        calculate_dgcforce_dparameters(l+ num_elements + element) = &
          unraveled_denforce_dparameters(element)%jii
      end do
!     deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + en_no_layers_of_elements(element - 1)
        end if
        do j = 1, en_no_layers_of_elements(element) - 1
          num_rows = en_no_nodes_of_elements(k + j) + 1
          num_cols = en_no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%en_weights(j)%twodarray)
          deallocate(unraveled_denforce_dparameters(&
          element)%en_weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%en_weights)
        deallocate(unraveled_denforce_dparameters(element)%en_weights)
      end do

      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + charge_no_layers_of_elements(element - 1)
        end if
        do j = 1, charge_no_layers_of_elements(element) - 1
          num_rows = charge_no_nodes_of_elements(k + j) + 1
          num_cols = charge_no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_charge_parameters(&
          element)%weights(j)%twodarray)
          deallocate(unraveled_denforce_dparameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_charge_parameters(element)%weights)
        deallocate(unraveled_denforce_dparameters(element)%weights)
      end do
      end function calculate_dgcforce_dparameters

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      end module chargeneuralnetwork

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
