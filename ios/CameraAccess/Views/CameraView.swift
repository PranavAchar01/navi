/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

//
// CameraView.swift
//
// The capture screen: a full-bleed camera preview with the belt, the latest answer on
// a solid card, the ask field and Preview / End session over it. Session, stream and
// server state, the server address, beep volume and Disconnect live in the settings
// sheet behind the gear; a single status dot stays on screen.
//

import MWDATCore
import SwiftUI
import UIKit

private let updateRequiredTitle = "Update required"

struct CameraView: View {
  @State private var viewModel: CameraViewModel
  @Bindable var wearablesVM: WearablesViewModel
  @Binding var source: CameraSource
  @State private var showSettings: Bool = false
  @State private var isLaunchingUpdate: Bool = false
  @State private var question: String = ""
  #if DEBUG
    @State private var displayPreviewLarge = DemoLaunch.largeDisplayPreview
  #endif
  @FocusState private var focusedField: Field?

  private enum Field {
    case question
  }

  init(wearables: WearablesInterface, wearablesVM: WearablesViewModel, link: ServerLink, source: Binding<CameraSource>) {
    self._viewModel = State(wrappedValue: CameraViewModel(wearables: wearables, link: link))
    self._wearablesVM = Bindable(wearablesVM)
    self._source = source
  }

  private var isUpdateRequired: Bool {
    wearablesVM.requiresFirmwareUpdate
  }

  /// The enlarged glasses-display preview (debug builds) takes the belt's place.
  private var beltHiddenForPreview: Bool {
    #if DEBUG
      return displayPreviewLarge && DemoLaunch.glasses != nil
    #else
      return false
    #endif
  }

  /// Whether the preview is live or tearing down — drives the in-row Preview
  /// toggle's icon, id, and action. (Recording implies a live preview.)
  private var previewIsActive: Bool {
    viewModel.isStreaming || viewModel.isRecording || viewModel.streamState == .stopping
  }

  var body: some View {
    ZStack {
      // Full-bleed preview (or a dark placeholder) fills the screen so overlaying
      // the controls never reflows it.
      previewBackground
        .edgesIgnoringSafeArea(.all)

      if let preview = viewModel.activePreview {
        CapturePreviewView(
          preview: preview,
          onDismiss: { viewModel.dismissCapturePreview() }
        )
        .transition(.opacity)
        .zIndex(1)
      }

      VStack(spacing: 0) {
        SourcePicker(source: $source)
          .environment(\.colorScheme, .dark)
          .padding(.horizontal, 20)
          .padding(.top, 8)
        topBar
          .padding(.horizontal, 20)
          .padding(.top, 12)
          .padding(.bottom, 40)
          .background(
            LinearGradient(colors: [.black.opacity(0.6), .clear], startPoint: .top, endPoint: .bottom)
              .edgesIgnoringSafeArea(.top)
          )

        HStack(alignment: .top) {
          #if DEBUG
            if let glasses = DemoLaunch.glasses {
              // Tap to enlarge the glasses' display for a closer look.
              let side: CGFloat = displayPreviewLarge ? 360 : 180
              MockDisplayPreview(glasses: glasses)
                .frame(width: side, height: side)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .onTapGesture { displayPreviewLarge.toggle() }
                .accessibilityIdentifier("glasses_display_preview")
            }
          #endif
          Spacer()
          if !beltHiddenForPreview {
            BeltView(belt: viewModel.link.belt, guideBearing: viewModel.link.guide?.bearing, cues: viewModel.link.proximity, warning: viewModel.link.warning)
              .frame(width: 150, height: 150)
              .background(Color.black.opacity(0.6))
              .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
              .accessibilityIdentifier("belt_view")
          }
        }
        .padding(.horizontal, 16)
        .padding(.top, -24)

        Spacer(minLength: 0)

        bottomBar
      }
    }
    .onAppear {
      viewModel.link.start()
      WatchLink.shared.activate()
      // Keep the screen on: backgrounding the app ends the glasses session.
      UIApplication.shared.isIdleTimerDisabled = true
    }
    .onDisappear {
      UIApplication.shared.isIdleTimerDisabled = false
      // Switching to the iPhone camera: let go of the glasses.
      if viewModel.hasSession { viewModel.endSession() }
    }
    #if DEBUG
    .task {
      await DemoLaunch.autoStart(viewModel)
      showSettings = DemoLaunch.opensSettings
    }
    #endif
    .animation(.spring(response: 0.42, dampingFraction: 0.86), value: viewModel.hasSession)
    .animation(.easeInOut(duration: 0.2), value: previewIsActive)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isStreaming)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isPaused)
    .animation(.easeInOut(duration: 0.2), value: viewModel.isRecording)
    .sheet(isPresented: $showSettings) {
      SettingsSheet(viewModel: viewModel, wearablesVM: wearablesVM)
    }
    .alert("Something went wrong", isPresented: $viewModel.showError) {
      Button("OK") { viewModel.dismissError() }
    } message: {
      Text(viewModel.errorMessage)
    }
    .alert("Allow camera access", isPresented: $viewModel.showCameraPermissionRedirectConfirm) {
      Button("Continue") { Task { await viewModel.confirmCameraPermissionRedirect() } }
      Button("Cancel", role: .cancel) {}
    } message: {
      Text("To preview your glasses camera, you'll be taken to the Meta AI app to grant access, then returned here.")
    }
    .navigationBarHidden(true)
  }

  // MARK: - Preview background

  @ViewBuilder
  private var previewBackground: some View {
    ZStack {
      Color.black
      if viewModel.showsLivePreview {
        // Live preview, the frozen last frame while paused, or the last frame held
        // while the stream tears down — so the paused badge / stop loader overlays
        // the preview rather than the start-streaming screen.
        LivePreviewView(viewModel: viewModel)
        if viewModel.isPaused {
          // Dim the frozen frame so the badge reads as intentionally paused.
          Color.black.opacity(0.35)
          pausedOverlay
        }
      } else if !viewModel.isBusy {
        statusPlaceholder
      }

      // Loader overlays whatever is behind it during any in-flight transition.
      if viewModel.isBusy {
        loadingOverlay
      }
    }
  }

  /// Label over the frozen preview while paused, so a held frame reads as
  /// intentionally paused (with how to resume) rather than a stalled feed. A
  /// `pause.fill` status glyph (not a circled/button shape) — resume is a glasses
  /// gesture, not an on-screen control.
  private var pausedOverlay: some View {
    VStack(spacing: 8) {
      Image(systemName: "pause.fill")
        .font(.system(size: 36))
        .foregroundStyle(.white)
      Text("Paused")
        .font(.system(size: 20, weight: .semibold))
        .foregroundStyle(.white)
      Text("Tap your glasses to resume.")
        .font(.system(size: 15))
        .multilineTextAlignment(.center)
        .foregroundStyle(Color.white.opacity(0.7))
    }
    .padding(.horizontal, 24)
    .accessibilityIdentifier("paused_overlay")
  }

  /// Dimmed spinner shown over the preview (or placeholder) while a session/stream
  /// start or stop is in flight.
  private var loadingOverlay: some View {
    ZStack {
      Color.black.opacity(0.35)
      ProgressView()
        .scaleEffect(1.5)
        .tint(.white)
    }
  }

  @ViewBuilder
  private var statusPlaceholder: some View {
    if !viewModel.hasActiveDevice {
      placeholder(title: "Put on your glasses", subtitle: nil, showWaitingRow: true)
    } else if isUpdateRequired {
      placeholder(title: updateRequiredTitle, subtitle: "Apply the update to start your camera.", showWaitingRow: false)
    } else if !viewModel.hasSession {
      placeholder(title: "Ready", subtitle: "Start a session to connect to your glasses.", showWaitingRow: false)
    } else if viewModel.isPaused {
      placeholder(title: "Paused", subtitle: "Tap your glasses to resume.", showWaitingRow: false)
    } else {
      placeholder(title: "Session started", subtitle: "Start the preview to see the live camera feed.", showWaitingRow: false)
    }
  }

  private func placeholder(title: String, subtitle: String?, showWaitingRow: Bool) -> some View {
    VStack(spacing: 12) {
      Image(.cameraAccessIcon)
        .resizable()
        .renderingMode(.template)
        .foregroundStyle(.white)
        .aspectRatio(contentMode: .fit)
        .frame(width: 88)

      Text(title)
        .font(.system(size: 20, weight: .semibold))
        .foregroundStyle(.white)

      if let subtitle {
        Text(subtitle)
          .font(.system(size: 15))
          .multilineTextAlignment(.center)
          .foregroundStyle(Color.white.opacity(0.7))
      }

      if showWaitingRow {
        HStack(spacing: 8) {
          Image(systemName: "hourglass")
            .resizable()
            .aspectRatio(contentMode: .fit)
            .foregroundStyle(Color.white.opacity(0.7))
            .frame(width: 16, height: 16)
          Text("Waiting for an active device")
            .font(.system(size: 14))
            .foregroundStyle(Color.white.opacity(0.7))
        }
      }
    }
    .padding(.horizontal, 24)
  }

  // MARK: - Top bar

  /// One status dot (green when the glasses stream to a connected server) and the gear.
  private var topBar: some View {
    HStack(alignment: .center, spacing: 12) {
      HStack(spacing: 6) {
        Circle()
          .fill(liveStatus.color)
          .frame(width: 8, height: 8)
        Text(liveStatus.text)
          .font(.system(size: 13, weight: .semibold))
          .foregroundStyle(.white)
      }
      .padding(.horizontal, 10)
      .padding(.vertical, 5)
      .background(.black.opacity(0.55), in: Capsule())
      .accessibilityIdentifier("server_chip")

      Spacer()

      iconButton("gearshape", label: "Settings", id: "settings_button") {
        showSettings = true
      }
    }
  }

  /// "Live" when frames are going to a connected server, else the first thing that isn't.
  private var liveStatus: (text: String, color: Color) {
    if viewModel.link.connection != .connected {
      return ("Server \(viewModel.link.connection.text)", .yellow)
    }
    if !viewModel.hasSession { return ("No session", .gray) }
    if !viewModel.isStreaming { return ("Camera \(viewModel.streamStateText)", .yellow) }
    return ("Live", .green)
  }

  private func iconButton(_ systemName: String, label: String, id: String, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      Image(systemName: systemName)
        .resizable()
        .aspectRatio(contentMode: .fit)
        .foregroundStyle(.white)
        .frame(width: 22, height: 22)
        .padding(.leading, 16)
    }
    .accessibilityLabel(label)
    .accessibilityIdentifier(id)
  }

  // MARK: - Bottom bar

  private var bottomBar: some View {
    VStack(spacing: 14) {
      if isUpdateRequired {
        updateControls
      } else {
        metaSightPanel
        // Reserved capture-row space + a single persistent button hold the button at
        // a fixed Y across every state.
        captureRow
        anchoredPrimaryButton
      }
    }
    .frame(maxWidth: .infinity)
    .padding(.horizontal, 24)
    .padding(.top, 48)
    .padding(.bottom, 28)
    .background(
      LinearGradient(colors: [.clear, .black.opacity(0.75)], startPoint: .top, endPoint: .bottom)
        .edgesIgnoringSafeArea(.bottom)
    )
  }

  /// In-session toolbar (Preview / Photo), kept laid out so the
  /// session button below holds a fixed Y. Hidden without a session; Preview leads as
  /// the gateway and is the lone enabled control in session-ready, the rest enabling
  /// once streaming.
  private var captureRow: some View {
    HStack(spacing: 10) {
      previewPill
      photoButton
    }
    .opacity(viewModel.hasSession ? 1 : 0)
  }

  // MARK: - meta-sight panel

  /// The latest answer and the guide/path line on a solid card, then the ask field.
  private var metaSightPanel: some View {
    VStack(alignment: .leading, spacing: 10) {
      VStack(alignment: .leading, spacing: 6) {
        RecentAnswer(note: viewModel.link.answerNote, font: .system(size: 18, weight: .semibold))
        Text(viewModel.link.statusLine)
          .font(.system(size: 14, weight: .medium))
          .foregroundStyle(Color.white.opacity(viewModel.link.lastAnswer == nil ? 0.95 : 0.65))
          .lineLimit(2)
          .accessibilityIdentifier("status_line")
        if !viewModel.sensors.heard.isEmpty {
          Text("Heard: \(viewModel.sensors.heard)")
            .font(.system(size: 13))
            .foregroundStyle(Color.white.opacity(0.6))
            .lineLimit(1)
        }
      }
      .padding(14)
      .frame(maxWidth: .infinity, alignment: .leading)
      .background(Color(white: 0.09).opacity(0.94), in: RoundedRectangle(cornerRadius: 16, style: .continuous))

      HStack(spacing: 8) {
        TextField("where is my water bottle", text: $question)
          .submitLabel(.send)
          .focused($focusedField, equals: .question)
          .onSubmit(sendQuestion)
          .panelField()
          .accessibilityLabel("Question")
          .accessibilityIdentifier("ask_field")
        Button("Ask", action: sendQuestion)
          .panelButton()
          .disabled(viewModel.link.connection != .connected)
          .accessibilityIdentifier("ask_button")
      }
    }
    .frame(maxWidth: .infinity, alignment: .leading)
  }

  private func sendQuestion() {
    viewModel.link.ask(question)
    question = ""
    focusedField = nil
  }

  /// Photo needs a live stream.
  private var captureControlsEnabled: Bool { viewModel.isStreaming }

  /// Photo capture as a compact icon (no label) so the Preview and Record pills keep
  /// room for text. Enabled while streaming, including during a recording.
  private var photoButton: some View {
    Button {
      viewModel.capturePhoto()
    } label: {
      Image(systemName: "camera.fill")
        .font(.system(size: 16, weight: .semibold))
        .foregroundStyle(.white)
        .frame(width: 50, height: 50)
        .background(.ultraThinMaterial)
        .clipShape(Circle())
    }
    .accessibilityLabel("Take photo")
    .accessibilityIdentifier("capture_button")
    .disabled(!captureControlsEnabled)
    .opacity(captureControlsEnabled ? 1 : 0.35)
  }

  /// Preview on/off — the gateway for capture. The eye icon flips and dims when off;
  /// locked mid-recording so a recording can't lose its feed.
  private var previewPill: some View {
    capturePill(
      icon: previewIsActive ? "eye.fill" : "eye.slash.fill",
      title: "Preview",
      contentTint: .white,
      id: previewIsActive ? "stop_preview_button" : "start_preview_button"
    ) {
      if previewIsActive {
        viewModel.stopStreaming()
      } else {
        Task { await viewModel.startStreaming() }
      }
    }
    .disabled(previewToggleDisabled)
    .opacity(previewToggleDisabled ? 0.35 : 1)
  }

  private var previewToggleDisabled: Bool {
    if previewIsActive {
      return viewModel.isRecording || viewModel.isBusy
    }
    return !viewModel.isSessionActive || viewModel.isBusy
  }

  /// The single full-width session button: Start Session, or End Session
  /// (destructive) once a session exists. One instance, never branched, so it never
  /// remounts and holds a fixed Y — only its title / style / id / action change.
  private var anchoredPrimaryButton: some View {
    CustomButton(
      title: viewModel.hasSession ? "End session" : "Start session",
      style: viewModel.hasSession ? .destructive : .primary,
      isDisabled: primaryDisabled
    ) {
      if viewModel.hasSession {
        viewModel.endSession()
      } else {
        viewModel.startSession()
      }
    }
    .accessibilityIdentifier(viewModel.hasSession ? "end_session_button" : "start_session_button")
  }

  /// Start needs an active device; End stays available mid-stream/record — the SDK
  /// cascades the stop, so only an in-flight transition disables it.
  private var primaryDisabled: Bool {
    if viewModel.hasSession {
      return viewModel.isBusy
    }
    return viewModel.isBusy || !viewModel.hasActiveDevice
  }

  private func primaryButton(_ title: String, id: String, disabled: Bool = false, action: @escaping () -> Void) -> some View {
    CustomButton(title: title, style: .primary, isDisabled: disabled, action: action)
      .accessibilityIdentifier(id)
  }

  /// Frosted icon+label capsule for the capture actions. `contentTint` colors both
  /// the icon and label so on/off dims them together (see `previewPill`).
  private func capturePill(icon: String, title: String, contentTint: Color = .white, id: String, action: @escaping () -> Void) -> some View {
    Button(action: action) {
      HStack(spacing: 8) {
        Image(systemName: icon)
          .font(.system(size: 15, weight: .semibold))
          .foregroundStyle(contentTint)
        Text(title)
          .font(.system(size: 15, weight: .semibold))
          .foregroundStyle(contentTint)
          .lineLimit(1)
          .minimumScaleFactor(0.85)
      }
      .frame(maxWidth: .infinity)
      .frame(height: 50)
      .background(.ultraThinMaterial)
      .clipShape(Capsule())
    }
    .accessibilityIdentifier(id)
  }

  // MARK: - Update controls

  @ViewBuilder
  private var updateControls: some View {
    UpdateRequiredMessage()
    // Gate the button while the external update flow launches so a double-tap can't
    // fire it twice; re-enabled once the await returns.
    primaryButton("Update firmware", id: "update_firmware_button", disabled: isLaunchingUpdate) {
      isLaunchingUpdate = true
      Task {
        await wearablesVM.openFirmwareUpdate()
        isLaunchingUpdate = false
      }
    }
  }
}

// MARK: - Live preview

/// The streaming preview image. Isolated from `CameraView` so frequent
/// (~24fps) `currentVideoFrame` updates re-render only this view, not the
/// surrounding controls.
private struct LivePreviewView: View {
  var viewModel: CameraViewModel

  var body: some View {
    if let videoFrame = viewModel.currentVideoFrame, viewModel.hasReceivedFirstFrame {
      GeometryReader { geometry in
        Image(uiImage: videoFrame)
          .resizable()
          .aspectRatio(contentMode: .fill)
          .frame(width: geometry.size.width, height: geometry.size.height)
          .clipped()
      }
    } else {
      ProgressView()
        .scaleEffect(1.5)
        .tint(.white)
    }
  }
}

// MARK: - Update banner

struct UpdateRequiredMessage: View {
  private let message = "Your glasses need a firmware update before Camera Access can start."

  var body: some View {
    HStack(alignment: .top, spacing: 12) {
      Image(systemName: "exclamationmark.triangle.fill")
        .resizable()
        .aspectRatio(contentMode: .fit)
        .foregroundStyle(Color.updateRequiredForeground)
        .frame(width: 24, height: 24)
        .accessibilityHidden(true)

      VStack(alignment: .leading, spacing: 4) {
        Text(updateRequiredTitle)
          .font(.system(size: 16, weight: .semibold))
          .foregroundStyle(Color.updateRequiredForeground)

        Text(message)
          .font(.system(size: 15))
          .foregroundStyle(Color.updateRequiredForeground)
          .fixedSize(horizontal: false, vertical: true)
      }

      Spacer(minLength: 0)
    }
    .padding(.all, 16)
    .frame(maxWidth: .infinity, alignment: .leading)
    .background(Color.updateRequiredBackground)
    .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
  }
}

// MARK: - Settings

/// Everything a developer needs and a wearer doesn't: live SDK and server state, the
/// server address, beep volume, voice permission and Disconnect.
private struct SettingsSheet: View {
  var viewModel: CameraViewModel
  @Bindable var wearablesVM: WearablesViewModel
  @Environment(\.dismiss) private var dismiss

  var body: some View {
    let link = viewModel.link
    NavigationStack {
      Form {
        Section("Status") {
          row("Session", viewModel.sessionStateText)
          row("Stream", viewModel.streamStateText)
          row("Server", link.connection.text)
          row("Glasses display", viewModel.glassesDisplay.status)
          row("Motion", viewModel.sensors.motionStatus)
          row("Voice", viewModel.sensors.voiceStatus)
          if let ms = link.belt.latencyMs {
            row("Belt latency", String(format: "%.0f ms", ms))
          }
        }
        Section("Server") {
          TextField("ws://host:8445/ws", text: Binding(get: { link.urlString }, set: { link.urlString = $0 }))
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .keyboardType(.URL)
            .submitLabel(.go)
            .onSubmit { link.applyURL() }
            .accessibilityLabel("Server address")
            .accessibilityIdentifier("server_url_field")
          Button("Connect") { link.applyURL() }
        }
        Section("Beeps") {
          Picker("Beeps", selection: Binding(get: { link.beeper.volume }, set: { link.beeper.volume = $0 })) {
            ForEach(BeepVolume.allCases) { Text($0.label).tag($0) }
          }
          .pickerStyle(.segmented)
          .accessibilityLabel("Beep volume")
          .accessibilityIdentifier("beep_volume")
        }
        if viewModel.sensors.voiceNeedsPermission {
          Section("Voice") {
            Button("Enable voice") {
              Task { await viewModel.sensors.requestVoicePermission() }
            }
          }
        }
        Section {
          Button("Disconnect glasses", role: .destructive) {
            wearablesVM.disconnectGlasses()
            dismiss()
          }
          .disabled(wearablesVM.registrationState != .registered)
        }
      }
      .navigationTitle("Settings")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } }
      }
    }
    .presentationDetents([.medium, .large])
    .environment(\.colorScheme, .dark)
  }

  private func row(_ label: String, _ value: String) -> some View {
    LabeledContent(label) {
      Text(value).monospaced()
    }
  }
}

// MARK: - meta-sight panel styling

extension View {
  func panelField() -> some View {
    self
      .font(.system(size: 14))
      .foregroundStyle(.white)
      .padding(.horizontal, 12)
      .frame(height: 38)
      .background(.ultraThinMaterial)
      .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
  }

  func panelButton() -> some View {
    self
      .font(.system(size: 14, weight: .semibold))
      .foregroundStyle(.white)
      .padding(.horizontal, 14)
      .frame(height: 38)
      .background(.ultraThinMaterial)
      .clipShape(Capsule())
  }
}

/// The latest answer, up for as long as the lens shows it (12 s), then folded to a
/// "Last answer · laptop" row; tapping it opens the answer again for another 12 s.
struct RecentAnswer: View {
  let note: AnswerNote?
  var font: Font
  @State private var reopenedAt: Date?

  var body: some View {
    if let note {
      TimelineView(.periodic(from: .now, by: 1)) { context in
        let open = [note.at, reopenedAt].compactMap { $0 }.contains {
          context.date.timeIntervalSince($0) < DisplayCard.answerSeconds
        }
        if open {
          Text(note.text)
            .font(font)
            .foregroundStyle(.white)
            .lineLimit(3)
            .fixedSize(horizontal: false, vertical: true)
            .onTapGesture { reopenedAt = nil }
            .accessibilityLabel("Answer: \(note.text)")
            .accessibilityIdentifier("answer_line")
        } else {
          Button {
            reopenedAt = Date()
          } label: {
            Label("Last answer · \(note.label ?? "show again")", systemImage: "arrow.uturn.backward")
              .font(.system(size: 14, weight: .semibold))
              .foregroundStyle(Color.white.opacity(0.75))
          }
          .buttonStyle(.plain)
          .accessibilityHint("Shows the last answer again")
          .accessibilityIdentifier("last_answer")
        }
      }
    }
  }
}
