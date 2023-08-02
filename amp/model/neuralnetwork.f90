!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     module that utilizes the regression model to calculate energies
!     and forces as well as their derivatives. Function names ending
!     with an underscore correspond to image-centered mode.

      module neuralnetwork
      implicit none

!     the data of neuralnetwork (should be fed in by python)
      double precision, allocatable::min_fingerprints(:, :)
      double precision, allocatable::max_fingerprints(:, :)
      integer, allocatable:: no_layers_of_elements(:)
      integer, allocatable:: no_nodes_of_elements(:)
      integer:: activation_signal

      type:: real_two_d_array
        sequence
        double precision, allocatable:: twodarray(:,:)
      end type real_two_d_array

      type:: element_parameters
        sequence
        double precision:: intercept
        double precision:: slope
        type(real_two_d_array), allocatable:: weights(:)
      end type element_parameters

      type:: real_one_d_array
        sequence
        double precision, allocatable:: onedarray(:)
      end type real_one_d_array

      contains

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the image-centered mode.
      function calculate_image_energy(num_inputs, inputs, num_parameters, &
      parameters)
      implicit none

      integer:: num_inputs, num_parameters
      double precision:: inputs(num_inputs)
      double precision:: parameters(num_parameters)
      double precision:: calculate_image_energy

      integer:: p, m, n, layer
      integer:: l, j, num_rows, num_cols, q
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope

!     changing the form of parameters from vector into derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(num_rows, num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  weights(j)%twodarray(p, q) = &
                  parameters(l + (p - 1) * num_cols + q)
              end do
          end do
          l = l + num_rows * num_cols
      end do
      intercept = parameters(l + 1)
      slope = parameters(l + 2)

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do

      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(num_inputs))
      allocate(ohat(1)%onedarray(num_inputs + 1))
      do m = 1, num_inputs
          o(1)%onedarray(m) = inputs(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) &
                  * weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = &
                  tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      calculate_image_energy = slope * o(layer)%onedarray(1) + intercept

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

!     deallocating derived type parameters
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)

      end function calculate_image_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns energy value in the atom-centered mode.
      function calculate_atomic_energy(symbol, &
      len_of_fingerprint, fingerprint, &
      num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, num_parameters, &
      len_of_fingerprint, num_elements
      double precision:: fingerprint(len_of_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_atomic_energy

      integer:: p, element, m, n, layer
      integer:: k, l, j, num_rows, num_cols, q
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(element_parameters):: unraveled_parameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - min_fingerprints(element, l)) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

!     changing the form of parameters from vector into derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
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
              p = p + no_layers_of_elements(element)
          end if
      end do
      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do

      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
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
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      calculate_atomic_energy = unraveled_parameters(element)%slope * &
      o(layer)%onedarray(1) + unraveled_parameters(element)%intercept

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

!      deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
          num_rows = no_nodes_of_elements(k + j) + 1
          num_cols = no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%weights)
      end do

      end function calculate_atomic_energy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns force value in the image-centered mode.
      function calculate_force_(num_inputs, inputs, inputs_, &
      num_parameters, parameters)
      implicit none

      integer:: num_inputs, num_parameters
      double precision:: inputs(num_inputs)
      double precision:: inputs_(num_inputs)
      double precision:: parameters(num_parameters)
      double precision:: calculate_force_

      double precision, allocatable:: temp(:)
      integer:: p, q, m, n, nn, layer
      integer:: l, j, num_rows, num_cols
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: doutputs_dinputs(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope

!     changing the form of parameters to derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(num_rows, num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  weights(j)%twodarray(p, q) = &
                  parameters(l + (p - 1) * num_cols + q)
              end do
          end do
          l = l + num_rows * num_cols
      end do

      intercept = parameters(l + 1)
      slope = parameters(l + 2)

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(num_inputs))
      allocate(ohat(1)%onedarray(num_inputs + 1))
      do m = 1, num_inputs
          o(1)%onedarray(m) = inputs(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * &
                  weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          deallocate(net)
      end do
      nn = size(o) - 2
      allocate(doutputs_dinputs(nn + 2))
      allocate(doutputs_dinputs(1)%onedarray(num_inputs))
      do m = 1, num_inputs
      doutputs_dinputs(1)%onedarray(m) = inputs_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(weights(layer)%twodarray, dim = 2)))
          do p = 1, size(weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                doutputs_dinputs(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      calculate_force_ = slope * doutputs_dinputs(nn + 2)%onedarray(1)
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      calculate_force_ = -1.0d0 * calculate_force_
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
      do p = 1, size(doutputs_dinputs)
          deallocate(doutputs_dinputs(p)%onedarray)
      end do
      deallocate(doutputs_dinputs)

!     deallocating derived type parameters
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)

      end function calculate_force_

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns force value in the atom-centered mode.
      function calculate_force(symbol, len_of_fingerprint, fingerprint, &
      fingerprintprime, num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, len_of_fingerprint, num_parameters
      integer:: num_elements
      double precision:: fingerprint(len_of_fingerprint)
      double precision:: fingerprintprime(len_of_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_force

      double precision, allocatable:: temp(:)
      integer:: p, q, element, m, n, nn, layer
      integer:: k, l, j, num_rows, num_cols
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: doutputs_dinputs(:)
      type(element_parameters):: unraveled_parameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)
      double precision:: fingerprintprime_(len_of_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - min_fingerprints(element, l)) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do
      ! scaling fingerprintprimes
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprintprime_(l) = &
            2.0d0 * fingerprintprime(l) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprintprime_(l) = fingerprintprime(l)
        endif
      end do

!     changing the form of parameters to derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
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
              p = p + no_layers_of_elements(element)
          end if
      end do

      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
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
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          deallocate(net)
      end do
      nn = size(o) - 2
      allocate(doutputs_dinputs(nn + 2))
      allocate(doutputs_dinputs(1)%onedarray(&
      len_of_fingerprint))
      do m = 1, len_of_fingerprint
      doutputs_dinputs(1)%onedarray(m) = fingerprintprime_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(unraveled_parameters(element)%weights(&
          layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_parameters(element)%weights(&
          layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_parameters(element)%weights(&
              layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_parameters(&
                  element)%weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
                doutputs_dinputs(layer+ 1)%onedarray(p) = temp(p)
              end if
          end do
          deallocate(temp)
      end do

      calculate_force = unraveled_parameters(element)%slope * &
      doutputs_dinputs(nn + 2)%onedarray(1)
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      calculate_force = -1.0d0 * calculate_force
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
      do p = 1, size(doutputs_dinputs)
          deallocate(doutputs_dinputs(p)%onedarray)
      end do
      deallocate(doutputs_dinputs)

!     deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
          num_rows = no_nodes_of_elements(k + j) + 1
          num_cols = no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%weights)
      end do

      end function calculate_force

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of energy w.r.t. parameters in the
!     image-centered mode.
      function calculate_denergy_dparameters_(num_inputs, inputs, &
      num_parameters, parameters)
      implicit none

      integer:: num_inputs, num_parameters
      double precision:: calculate_denergy_dparameters_(num_parameters)
      double precision:: parameters(num_parameters)
      double precision:: inputs(num_inputs)

      integer:: m, n, j, l, layer, p, q, nn, num_cols, num_rows
      double precision:: temp1, temp2
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope
      type(real_two_d_array), allocatable:: &
      unraveled_denergy_dweights(:)
      double precision:: denergy_dintercept
      double precision:: denergy_dslope

!     changing the form of parameters from vector into derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(num_rows, num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  weights(j)%twodarray(p, q) = &
                  parameters(l + (p - 1) * num_cols + q)
              end do
          end do
          l = l + num_rows * num_cols
      end do
      intercept = parameters(l + 1)
      slope = parameters(l + 2)

      denergy_dintercept = 0.d0
      denergy_dslope = 0.d0
      l = 0
      allocate(unraveled_denergy_dweights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(unraveled_denergy_dweights(j)%twodarray(num_rows, &
          num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  unraveled_denergy_dweights(j)%twodarray(p, q) = 0.0d0
              end do
          end do
          l = l + num_rows * num_cols
      end do

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(num_inputs))
      allocate(ohat(1)%onedarray(num_inputs + 1))
      do m = 1, num_inputs
          o(1)%onedarray(m) = inputs(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * weights(&
                  layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = &
                  (1.0d0 - o(layer + 1)%onedarray(j)* &
                  o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
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
                  weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      denergy_dintercept = 1.0d0
      denergy_dslope = o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  unraveled_denergy_dweights(layer)%twodarray(p, q) = &
                  slope * &
                  ohat(layer)%onedarray(p) * delta(layer)%onedarray(q)
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

!     changing the derivatives of the energy from derived-type
!     form into vector
      l = 0
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          do p = 1, num_rows
              do q = 1, num_cols
                  calculate_denergy_dparameters_(&
                  l + (p - 1) * num_cols + q) = &
                  unraveled_denergy_dweights(j)%twodarray(p, q)
              end do
          end do
          l = l + num_rows * num_cols
      end do

      calculate_denergy_dparameters_(l + 1) = denergy_dintercept
      calculate_denergy_dparameters_(l + 2) = denergy_dslope

!     deallocating derived-type parameters
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)
      do p = 1, size(unraveled_denergy_dweights)
          deallocate(unraveled_denergy_dweights(p)%twodarray)
      end do
      deallocate(unraveled_denergy_dweights)

      end function calculate_denergy_dparameters_

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of energy w.r.t. parameters in the
!     atom-centered mode.
      function calculate_datomicenergy_dparameters(symbol, &
      len_of_fingerprint, fingerprint, num_elements, &
      elements_numbers, num_parameters, parameters)
      implicit none

      integer:: num_parameters, num_elements
      integer:: symbol, len_of_fingerprint
      double precision:: calculate_datomicenergy_dparameters(num_parameters)
      double precision:: parameters(num_parameters)
      double precision:: fingerprint(len_of_fingerprint)
      integer:: elements_numbers(num_elements)

      integer:: element, m, n, j, k, l, layer, p, q, nn, num_cols
      integer:: num_rows
      double precision:: temp1, temp2
      integer, allocatable:: hiddensizes(:)
      double precision, allocatable:: net(:)
      type(real_one_d_array), allocatable:: o(:), ohat(:)
      type(real_one_d_array), allocatable:: delta(:), D(:)
      type(element_parameters):: unraveled_parameters(num_elements)
      type(element_parameters):: &
      unraveled_daenergy_dparameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - min_fingerprints(element, l)) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do

!     changing the form of parameters to derived types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_parameters(element)%weights(&
            j)%twodarray(num_rows, num_cols))
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

      do element = 1, num_elements
        unraveled_daenergy_dparameters(element)%intercept = 0.d0
        unraveled_daenergy_dparameters(element)%slope = 0.d0
      end do
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_daenergy_dparameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_daenergy_dparameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_daenergy_dparameters(&
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
              p = p + no_layers_of_elements(element)
          end if
      end do
      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
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
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = (1.0d0 - &
                  o(layer + 1)%onedarray(j)* o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
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
                  element)%weights(layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      unraveled_daenergy_dparameters(element)%intercept = 1.0d0
      unraveled_daenergy_dparameters(element)%slope = &
      o(nn + 2)%onedarray(1)
      do layer = 1, nn + 1
          do p = 1, size(ohat(layer)%onedarray)
              do q = 1, size(delta(layer)%onedarray)
                  unraveled_daenergy_dparameters(element)%weights(&
                  layer)%twodarray(p, q) = &
                  unraveled_parameters(element)%slope * &
                  ohat(layer)%onedarray(p) * delta(layer)%onedarray(q)
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

!     changing the derivatives of the energy from derived-type
!     form into vector
      k = 0
      l = 0
      do element = 1, num_elements
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_datomicenergy_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_daenergy_dparameters(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_datomicenergy_dparameters(l + 2 *  element - 1) = &
        unraveled_daenergy_dparameters(element)%intercept
        calculate_datomicenergy_dparameters(l + 2 * element) = &
        unraveled_daenergy_dparameters(element)%slope
      end do

!     deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
          num_rows = no_nodes_of_elements(k + j) + 1
          num_cols = no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%weights(j)%twodarray)
          deallocate(unraveled_daenergy_dparameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%weights)
        deallocate(unraveled_daenergy_dparameters(element)%weights)
      end do

      end function calculate_datomicenergy_dparameters

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of force w.r.t. parameters in the
!     image-centered mode.
      function calculate_dforce_dparameters_(num_inputs, inputs, &
      inputs_, num_parameters, parameters)
      implicit none

      integer:: num_inputs, num_parameters
      double precision:: calculate_dforce_dparameters_(num_parameters)
      double precision:: parameters(num_parameters)
      double precision:: inputs(num_inputs)
      double precision:: inputs_(num_inputs)

      integer:: m, n, j, l, layer, p, q, nn, num_cols
      integer:: num_rows
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
      double precision, allocatable:: temp(:), temp3(:), temp4(:)
      double precision, allocatable:: temp5(:), temp6(:)
      type(real_two_d_array), allocatable:: weights(:)
      double precision:: intercept
      double precision:: slope
      type(real_two_d_array), allocatable:: unraveled_dforce_dweights(:)
      double precision:: dforce_dintercept
      double precision:: dforce_dslope

!     changing the form of parameters from vector into derived-types
      l = 0
      allocate(weights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(weights(j)%twodarray(num_rows, num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  weights(j)%twodarray(p, q) = &
                  parameters(l + (p - 1) * num_cols + q)
              end do
          end do
          l = l + num_rows * num_cols
      end do
      intercept = parameters(l + 1)
      slope = parameters(l + 2)

      dforce_dintercept = 0.d0
      dforce_dslope = 0.d0
      l = 0
      allocate(unraveled_dforce_dweights(no_layers_of_elements(1)-1))
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          allocate(unraveled_dforce_dweights(j)%twodarray(num_rows, &
          num_cols))
          do p = 1, num_rows
              do q = 1, num_cols
                  unraveled_dforce_dweights(j)%twodarray(p, q) = 0.0d0
              end do
          end do
          l = l + num_rows * num_cols
      end do

      allocate(hiddensizes(no_layers_of_elements(1) - 2))
      do m = 1, no_layers_of_elements(1) - 2
          hiddensizes(m) = no_nodes_of_elements(m + 1)
      end do
      allocate(o(no_layers_of_elements(1)))
      allocate(ohat(no_layers_of_elements(1)))
      layer = 1
      allocate(o(1)%onedarray(num_inputs))
      allocate(ohat(1)%onedarray(num_inputs + 1))
      do m = 1, num_inputs
          o(1)%onedarray(m) = inputs(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(weights(layer)%twodarray, dim=1) - 1
              ohat(layer)%onedarray(m) = o(layer)%onedarray(m)
          end do
          ohat(layer)%onedarray(&
          size(weights(layer)%twodarray, dim=1)) = 1.0d0
          allocate(net(size(weights(layer)%twodarray, dim=2)))
          allocate(o(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2)))
          allocate(ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1))
          do m = 1, size(weights(layer)%twodarray, dim=2)
              net(m) = 0.0d0
              do n = 1, size(weights(layer)%twodarray, dim=1)
                  net(m) =  net(m) + &
                  ohat(layer)%onedarray(n) * &
                  weights(layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(&
          size(weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      nn = size(o) - 2
      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = (1.0d0 - &
                  o(layer + 1)%onedarray(j)* o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
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
                  temp1 = D(layer)%onedarray(p) * weights(&
                  layer + 1)%twodarray(p, q)
                  temp2 = temp1  * delta(layer + 1)%onedarray(q)
                  delta(layer)%onedarray(p) = &
                  delta(layer)%onedarray(p) + temp2
              end do
          end do
      end do

      allocate(doutputs_dinputs(nn + 2))
      allocate(doutputs_dinputs(1)%onedarray(num_inputs))
      do m = 1, num_inputs
        doutputs_dinputs(1)%onedarray(m) = inputs_(m)
      end do
      do layer = 1, nn + 1
          allocate(temp(size(weights(layer)%twodarray, dim = 2)))
          do p = 1, size(weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = &
                temp(p) * (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
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
              if (activation_signal == 1) then
                  dD_dinputs(layer)%onedarray(p) = &
                  - 2.0d0 * o(layer + 1)%onedarray(p) * &
                  doutputs_dinputs(layer + 1)%onedarray(p)
              elseif (activation_signal == 2) then
                  dD_dinputs(layer)%onedarray(p) = &
                  doutputs_dinputs(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * o(layer + 1)%onedarray(p))
              elseif (activation_signal == 3) then
                  dD_dinputs(layer)%onedarray(p) =0.0d0
              end if
          end do
      end do

      allocate(ddelta_dinputs(nn + 1))
      allocate(ddelta_dinputs(nn + 1)%onedarray(1))
      ddelta_dinputs(nn + 1)%onedarray(1) = &
      dD_dinputs(nn + 1)%onedarray(1)

      do layer = nn, 1, -1
          allocate(temp3(&
          size(weights(layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(&
          size(weights(layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(weights(layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + weights(layer + 1)%twodarray(&
                  p, q) * delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + weights(layer + 1)%twodarray(&
                  p, q) * ddelta_dinputs(layer + 1)%onedarray(q)
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

      dforce_dslope = doutputs_dinputs(nn + 2)%onedarray(1)
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      dforce_dslope = -1.0d0 * dforce_dslope
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
              unraveled_dforce_dweights(layer)%twodarray(p, q) = &
              slope * doutput_dinputsdweights(p, q)
              ! force is multiplied by -1, because it is -dE/dx and
              ! not dE/dx.
                unraveled_dforce_dweights(layer)%twodarray(p, q) = &
               -1.0d0 * unraveled_dforce_dweights(layer)%twodarray(p, q)
              end do
          end do
          deallocate(dohat_dinputs)
          deallocate(doutput_dinputsdweights)
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

      l = 0
      do j = 1, no_layers_of_elements(1) - 1
          num_rows = no_nodes_of_elements(j) + 1
          num_cols = no_nodes_of_elements(j + 1)
          do p = 1, num_rows
              do q = 1, num_cols
                  calculate_dforce_dparameters_(&
                  l + (p - 1) * num_cols + q) = &
                  unraveled_dforce_dweights(j)%twodarray(p, q)
              end do
          end do
          l = l + num_rows * num_cols
      end do
      calculate_dforce_dparameters_(l + 1) = dforce_dintercept
      calculate_dforce_dparameters_(l + 2) = dforce_dslope

!     deallocating derived-type parameters
      do p = 1, size(weights)
          deallocate(weights(p)%twodarray)
      end do
      deallocate(weights)
      do p = 1, size(unraveled_dforce_dweights)
          deallocate(unraveled_dforce_dweights(p)%twodarray)
      end do
      deallocate(unraveled_dforce_dweights)

      end function calculate_dforce_dparameters_

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!     Returns derivative of force w.r.t. parameters in the
!     atom-centered mode
      function calculate_dforce_dparameters(symbol, len_of_fingerprint, &
      fingerprint, fingerprintprime, num_elements, elements_numbers, &
      num_parameters, parameters)
      implicit none

      integer:: symbol, len_of_fingerprint
      integer:: num_parameters, num_elements
      double precision:: fingerprint(len_of_fingerprint)
      double precision:: fingerprintprime(len_of_fingerprint)
      integer:: elements_numbers(num_elements)
      double precision:: parameters(num_parameters)
      double precision:: calculate_dforce_dparameters(num_parameters)

      integer:: element, m, n, j, k, l, layer, p, q, nn, num_cols
      integer:: num_rows
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
      double precision, allocatable:: temp(:), temp3(:), temp4(:)
      double precision, allocatable:: temp5(:), temp6(:)
      type(element_parameters):: unraveled_parameters(num_elements)
      type(element_parameters):: &
      unraveled_dforce_dparameters(num_elements)
      double precision:: fingerprint_(len_of_fingerprint)
      double precision:: fingerprintprime_(len_of_fingerprint)

      ! scaling fingerprints
      do element = 1, num_elements
        if (symbol == &
            elements_numbers(element)) then
            exit
        end if
      end do
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprint_(l) = -1.0d0 + 2.0d0 * &
            (fingerprint(l) - min_fingerprints(element, l)) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprint_(l) = fingerprint(l)
        endif
      end do
      ! scaling fingerprintprimes
      do l = 1, len_of_fingerprint
        if ((max_fingerprints(element, l) - &
        min_fingerprints(element, l)) .GT. &
        (10.0d0 ** (-8.0d0))) then
            fingerprintprime_(l) = &
            2.0d0 * fingerprintprime(l) / &
            (max_fingerprints(element, l) - &
            min_fingerprints(element, l))
        else
            fingerprintprime_(l) = fingerprintprime(l)
        endif
      end do

!     changing the form of parameters from vector into derived-types
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_parameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element .GT. 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
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

      do element = 1, num_elements
        unraveled_dforce_dparameters(element)%intercept = 0.d0
        unraveled_dforce_dparameters(element)%slope = 0.d0
      end do
      k = 0
      l = 0
      do element = 1, num_elements
        allocate(unraveled_dforce_dparameters(element)%weights(&
        no_layers_of_elements(element)-1))
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
            allocate(unraveled_dforce_dparameters(&
            element)%weights(j)%twodarray(num_rows, num_cols))
            do p = 1, num_rows
                do q = 1, num_cols
                    unraveled_dforce_dparameters(&
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
              p = p + no_layers_of_elements(element)
          end if
      end do

      allocate(hiddensizes(no_layers_of_elements(element) - 2))
      do m = 1, no_layers_of_elements(element) - 2
          hiddensizes(m) = no_nodes_of_elements(p + m + 1)
      end do
      allocate(o(no_layers_of_elements(element)))
      allocate(ohat(no_layers_of_elements(element)))
      layer = 1
      allocate(o(1)%onedarray(len_of_fingerprint))
      allocate(ohat(1)%onedarray(len_of_fingerprint + 1))
      do m = 1, len_of_fingerprint
          o(1)%onedarray(m) = fingerprint_(m)
      end do
      do layer = 1, size(hiddensizes) + 1
          do m = 1, size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=1) - 1
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
                  ohat(layer)%onedarray(n) * &
                  unraveled_parameters(element)%weights(&
                  layer)%twodarray(n, m)
              end do
              if (activation_signal == 1) then
                  o(layer + 1)%onedarray(m) = tanh(net(m))
              else if (activation_signal == 2) then
                  o(layer + 1)%onedarray(m) = &
                  1.0d0 / (1.0d0 +  exp(- net(m)))
              else if (activation_signal == 3) then
                  o(layer + 1)%onedarray(m) = net(m)
              end if
              ohat(layer + 1)%onedarray(m) = o(layer + 1)%onedarray(m)
          end do
          ohat(layer + 1)%onedarray(size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim=2) + 1) =  1.0d0
          deallocate(net)
      end do

      nn = size(o) - 2

      allocate(D(nn + 1))
      do layer = 1, nn + 1
          allocate(D(layer)%onedarray(size(o(layer + 1)%onedarray)))
          do j = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                  D(layer)%onedarray(j) = &
                  (1.0d0 - o(layer + 1)%onedarray(j)* &
                  o(layer + 1)%onedarray(j))
              elseif (activation_signal == 2) then
                  D(layer)%onedarray(j) = o(layer + 1)%onedarray(j) * &
                  (1.0d0 - o(layer + 1)%onedarray(j))
              elseif (activation_signal == 3) then
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
                  unraveled_parameters(element)%weights(&
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
          element)%weights(layer)%twodarray, dim = 2)))
          do p = 1, size(unraveled_parameters(&
          element)%weights(layer)%twodarray, dim = 2)
              temp(p) = 0.0d0
              do q = 1, size(unraveled_parameters(&
              element)%weights(layer)%twodarray, dim = 1) - 1
                  temp(p) = temp(p) + doutputs_dinputs(&
                  layer)%onedarray(q) * unraveled_parameters(&
                  element)%weights(layer)%twodarray(q, p)
              end do
          end do
          q = size(o(layer + 1)%onedarray)
          allocate(doutputs_dinputs(layer + 1)%onedarray(q))
          do p = 1, size(o(layer + 1)%onedarray)
              if (activation_signal == 1) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p) * &
                o(layer + 1)%onedarray(p))
              else if (activation_signal == 2) then
                doutputs_dinputs(layer + 1)%onedarray(p) = temp(p) * &
                (1.0d0 - o(layer + 1)%onedarray(p)) * &
                o(layer + 1)%onedarray(p)
              else if (activation_signal == 3) then
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
              if (activation_signal == 1) then
                  dD_dinputs(layer)%onedarray(p) =- 2.0d0 * &
                  o(layer + 1)%onedarray(p) * &
                  doutputs_dinputs(layer + 1)%onedarray(p)
              elseif (activation_signal == 2) then
                  dD_dinputs(layer)%onedarray(p) = &
                  doutputs_dinputs(layer + 1)%onedarray(p) * &
                  (1.0d0 - 2.0d0 * o(layer + 1)%onedarray(p))
              elseif (activation_signal == 3) then
                  dD_dinputs(layer)%onedarray(p) =0.0d0
              end if
          end do
      end do

      allocate(ddelta_dinputs(nn + 1))
      allocate(ddelta_dinputs(nn + 1)%onedarray(1))
      ddelta_dinputs(nn + 1)%onedarray(1) = &
      dD_dinputs(nn + 1)%onedarray(1)

      do layer = nn, 1, -1
          allocate(temp3(size(unraveled_parameters(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          allocate(temp4(size(unraveled_parameters(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1))
          do p = 1, size(unraveled_parameters(element)%weights(&
          layer + 1)%twodarray, dim = 1) - 1
              temp3(p) = 0.0d0
              temp4(p) = 0.0d0
              do q = 1, size(delta(layer + 1)%onedarray)
                  temp3(p) = temp3(p) + unraveled_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
                  delta(layer + 1)%onedarray(q)
                  temp4(p) = temp4(p) + unraveled_parameters(&
                  element)%weights(layer + 1)%twodarray(p, q) * &
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

      unraveled_dforce_dparameters(element)%slope = &
      doutputs_dinputs(nn + 2)%onedarray(1)
      ! force is multiplied by -1, because it is -dE/dx and not dE/dx.
      unraveled_dforce_dparameters(element)%slope = &
      -1.0d0 * unraveled_dforce_dparameters(element)%slope
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
              unraveled_dforce_dparameters(element)%weights(&
              layer)%twodarray(p, q) = &
              unraveled_parameters(element)%slope * &
              doutput_dinputsdweights(p, q)
              ! force is multiplied by -1, because it is -dE/dx and
              ! not dE/dx.
              unraveled_dforce_dparameters(element)%weights(&
              layer)%twodarray(p, q) = &
              -1.0d0 * unraveled_dforce_dparameters(element)%weights(&
              layer)%twodarray(p, q)
              end do
          end do
          deallocate(dohat_dinputs)
          deallocate(doutput_dinputsdweights)
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

      k = 0
      l = 0
      do element = 1, num_elements
        if (element > 1) then
            k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
            num_rows = no_nodes_of_elements(k + j) + 1
            num_cols = no_nodes_of_elements(k + j + 1)
            do p = 1, num_rows
                do q = 1, num_cols
                    calculate_dforce_dparameters(&
                    l + (p - 1) * num_cols + q) = &
                    unraveled_dforce_dparameters(&
                    element)%weights(j)%twodarray(p, q)
                end do
            end do
            l = l + num_rows * num_cols
        end do
      end do
      do element = 1, num_elements
        calculate_dforce_dparameters(l + 2 *  element - 1) = &
        unraveled_dforce_dparameters(element)%intercept
        calculate_dforce_dparameters(l + 2 * element) = &
        unraveled_dforce_dparameters(element)%slope
      end do

!     deallocating derived-type parameters
      k = 0
      do element = 1, num_elements
        if (element .GT. 1) then
          k = k + no_layers_of_elements(element - 1)
        end if
        do j = 1, no_layers_of_elements(element) - 1
          num_rows = no_nodes_of_elements(k + j) + 1
          num_cols = no_nodes_of_elements(k + j + 1)
          deallocate(unraveled_parameters(&
          element)%weights(j)%twodarray)
          deallocate(unraveled_dforce_dparameters(&
          element)%weights(j)%twodarray)
        end do
        deallocate(unraveled_parameters(element)%weights)
        deallocate(unraveled_dforce_dparameters(element)%weights)
      end do

      end function calculate_dforce_dparameters

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      end module neuralnetwork

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
