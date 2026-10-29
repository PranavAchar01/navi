//
// BeltBLE.swift
//
// The belt over Bluetooth LE. The server sends each belt command to the app (`beltcmd`, base64 of
// the 26-byte packet in firmware/belt/belt_core.h); this writes it, unchanged, to the belt's
// "metasight-belt" GATT characteristic without response. The app scans for the belt's service,
// connects, and reconnects on its own when the belt drops or goes out of range. Background
// operation relies on the bluetooth-central background mode and state restoration. If the link
// is down the belt's own watchdog plays the lost-link taps within 250 ms, so a gap is never silent.
//
// UUIDs must match firmware/belt/belt.ino and metasight/belt.py. Prototype: not yet run against
// a real belt.
//

@preconcurrency import CoreBluetooth
import Foundation
import Observation

@MainActor
@Observable
final class BeltBLE: NSObject {
  enum Status: Equatable {
    case idle
    case bluetoothOff
    case scanning
    case connecting
    case connected

    var label: String {
      switch self {
      case .idle: "Belt: not started"
      case .bluetoothOff: "Belt: Bluetooth off"
      case .scanning: "Belt: searching"
      case .connecting: "Belt: connecting"
      case .connected: "Belt: connected"
      }
    }
  }

  static let service = CBUUID(string: "6d657461-7369-6768-742d-62656c740000")
  static let characteristic = CBUUID(string: "6d657461-7369-6768-742d-62656c740001")
  static let name = "metasight-belt"
  nonisolated static let packetLength = 26
  static let restoreID = "metasight.belt"

  private(set) var status: Status = .idle
  /// Packets written since the app started (for the settings chip and tests).
  private(set) var sent = 0

  @ObservationIgnored private var central: CBCentralManager?
  @ObservationIgnored private var peripheral: CBPeripheral?
  @ObservationIgnored private var command: CBCharacteristic?

  var isConnected: Bool { status == .connected }

  /// A valid command packet: 26 bytes starting "MS".
  nonisolated static func accepts(_ data: Data) -> Bool {
    data.count == packetLength && data.starts(with: [0x4D, 0x53])
  }

  /// Starts scanning; idempotent. Asks for the Bluetooth permission the first time.
  func start() {
    guard central == nil else { return }
    central = CBCentralManager(
      delegate: self, queue: .main,
      options: [CBCentralManagerOptionRestoreIdentifierKey: Self.restoreID])
  }

  /// Writes a packet if the belt is connected and can take it; otherwise drops it (the belt
  /// would rather have the next, fresher command than a queue of old ones).
  func write(_ packet: Data) {
    start()
    guard Self.accepts(packet), let peripheral, let command, status == .connected,
      peripheral.canSendWriteWithoutResponse,
      peripheral.maximumWriteValueLength(for: .withoutResponse) >= Self.packetLength
    else { return }
    peripheral.writeValue(packet, for: command, type: .withoutResponse)
    sent += 1
  }

  private func scan() {
    guard let central, central.state == .poweredOn else { return }
    status = .scanning
    central.scanForPeripherals(withServices: [Self.service])
  }

  private func drop() {
    command = nil
    if let peripheral { central?.cancelPeripheralConnection(peripheral) }
    peripheral = nil
    scan()
  }
}

extension BeltBLE: CBCentralManagerDelegate, CBPeripheralDelegate {
  nonisolated func centralManagerDidUpdateState(_ central: CBCentralManager) {
    MainActor.assumeIsolated {
      if central.state == .poweredOn {
        if let peripheral, peripheral.state == .connected {
          peripheral.discoverServices([Self.service])
        } else {
          scan()
        }
      } else {
        status = central.state == .poweredOff ? .bluetoothOff : .idle
        command = nil
      }
    }
  }

  nonisolated func centralManager(
    _ central: CBCentralManager, willRestoreState dict: [String: Any]
  ) {
    nonisolated(unsafe) let restored = dict[CBCentralManagerRestoredStatePeripheralsKey] as? [CBPeripheral]
    MainActor.assumeIsolated {
      if let p = restored?.first {
        peripheral = p
        p.delegate = self
      }
    }
  }

  nonisolated func centralManager(
    _ central: CBCentralManager, didDiscover peripheral: CBPeripheral,
    advertisementData: [String: Any], rssi RSSI: NSNumber
  ) {
    MainActor.assumeIsolated {
      central.stopScan()
      self.peripheral = peripheral
      peripheral.delegate = self
      status = .connecting
      central.connect(peripheral)
    }
  }

  nonisolated func centralManager(_ central: CBCentralManager, didConnect peripheral: CBPeripheral) {
    MainActor.assumeIsolated { peripheral.discoverServices([Self.service]) }
  }

  nonisolated func centralManager(
    _ central: CBCentralManager, didFailToConnect peripheral: CBPeripheral, error: Error?
  ) {
    MainActor.assumeIsolated { drop() }
  }

  nonisolated func centralManager(
    _ central: CBCentralManager, didDisconnectPeripheral peripheral: CBPeripheral, error: Error?
  ) {
    MainActor.assumeIsolated {
      command = nil
      self.peripheral = nil
      scan()
    }
  }

  nonisolated func peripheral(_ peripheral: CBPeripheral, didDiscoverServices error: Error?) {
    MainActor.assumeIsolated {
      guard let service = peripheral.services?.first(where: { $0.uuid == Self.service }) else {
        drop()
        return
      }
      peripheral.discoverCharacteristics([Self.characteristic], for: service)
    }
  }

  nonisolated func peripheral(
    _ peripheral: CBPeripheral, didDiscoverCharacteristicsFor service: CBService, error: Error?
  ) {
    MainActor.assumeIsolated {
      command = service.characteristics?.first { $0.uuid == Self.characteristic }
      status = command == nil ? .connecting : .connected
      if command == nil { drop() }
    }
  }
}
