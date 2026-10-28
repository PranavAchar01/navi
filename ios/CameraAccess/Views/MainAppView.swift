/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// MainAppView.swift
//
// Central navigation hub that displays different views based on DAT SDK registration and device states.
// When unregistered, shows the registration flow. When registered, shows the device selection screen
// for choosing which Meta wearable device to stream from.
//

import MWDATCore
import SwiftUI

struct MainAppView: View {
  let wearables: WearablesInterface
  var viewModel: WearablesViewModel
  /// One server connection for both camera sources, so the watch keeps its feed across a switch.
  let link: ServerLink
  @AppStorage(CameraSource.defaultsKey) private var source: CameraSource = .glasses

  var body: some View {
    switch source {
    case .iphone:
      // The phone's own camera needs no glasses registration.
      PhoneCameraView(link: link, source: $source)
    case .glasses:
      if viewModel.registrationState == .registered {
        CameraView(wearables: wearables, wearablesVM: viewModel, link: link, source: $source)
      } else {
        // User not registered - show registration/onboarding flow
        HomeScreenView(viewModel: viewModel, source: $source)
      }
    }
  }
}
