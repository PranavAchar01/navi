/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// HomeScreenView.swift
//
// Welcome screen that guides users through the DAT SDK registration process.
// This view is displayed when the app is not yet registered.
//

import MWDATCore
import SwiftUI

struct HomeScreenView: View {
  var viewModel: WearablesViewModel
  @Binding var source: CameraSource

  var body: some View {
    ZStack {
      Color.white.edgesIgnoringSafeArea(.all)

      VStack(spacing: 12) {
        SourcePicker(source: $source)

        Spacer()

        Image(.cameraAccessIcon)
          .resizable()
          .aspectRatio(contentMode: .fit)
          .frame(width: 120)

        VStack(spacing: 12) {
          TipRow(
            resource: .smartGlassesIcon,
            title: "Start a session",
            text: "Connect to your glasses to get started."
          )
          TipRow(
            resource: .videoIcon,
            title: "Start preview",
            text: "See the live camera feed from your glasses."
          )
          TipRow(
            resource: .tapIcon,
            title: "Capture",
            text: "Take photos and record video with audio from your glasses' microphone."
          )
        }

        Spacer()

        VStack(spacing: 20) {
          Text("You'll be redirected to the Meta AI app to confirm your connection.")
            .font(.system(size: 14))
            .foregroundStyle(.gray)
            .multilineTextAlignment(.center)
            .fixedSize(horizontal: false, vertical: true)
            .padding(.horizontal, 12)

          CustomButton(
            title: viewModel.registrationState == .registering ? "Connecting…" : "Connect my glasses",
            style: .primary,
            isDisabled: viewModel.registrationState == .registering
          ) {
            viewModel.connectGlasses()
          }

          Button("No glasses? Use the iPhone camera") {
            source = .iphone
          }
          .font(.system(size: 17, weight: .semibold))
          .frame(maxWidth: .infinity, minHeight: 44)
          .accessibilityHint("Switches to this iPhone's camera, no glasses needed")
          .accessibilityIdentifier("use_iphone_button")
        }
      }
      .padding(.all, 24)
    }
  }
}
