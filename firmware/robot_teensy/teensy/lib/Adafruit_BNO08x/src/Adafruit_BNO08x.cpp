/*!
 *  @file Adafruit_BNO08x.cpp
 *
 *  @mainpage Adafruit BNO08x 9-DOF Orientation IMU Fusion Breakout
 *
 *  @section intro_sec Introduction
 *
 * 	I2C Driver for the Library for the BNO08x 9-DOF Orientation IMU Fusion
 * Breakout
 *
 * 	This is a library for the Adafruit BNO08x breakout:
 * 	https://www.adafruit.com/product/4754
 *
 * 	Adafruit invests time and resources providing this open source code,
 *  please support Adafruit and open-source hardware by purchasing products from
 * 	Adafruit!
 *
 *  @section dependencies Dependencies
 *  This library depends on the Adafruit BusIO library
 *
 *  This library depends on the Adafruit Unified Sensor library
 *
 *  @section author Author
 *
 *  Bryan Siepert for Adafruit Industries
 *
 * 	@section license License
 *
 * 	BSD (see license.txt)
 *
 * 	@section  HISTORY
 *
 *     v1.0 - First release
 */

#include "Arduino.h"
#include <Wire.h>

#include "Adafruit_BNO08x.h"

static Adafruit_SPIDevice *spi_dev = NULL; ///< Pointer to SPI bus interface
static int8_t _int_pin, _reset_pin;

static Adafruit_I2CDevice *i2c_dev = NULL; ///< Pointer to I2C bus interface
static HardwareSerial *uart_dev = NULL;

// sh2_service() may decode several reports from one SHTP packet. Upstream's
// single output pointer was overwritten by every callback, so only the last
// report in that packet survived. With 400 Hz GRV/gyro alongside the 50 Hz
// linear-acceleration report, the supposedly 50 Hz acceleration stream reached
// the application at only ~2.5 Hz in LOG0015. Keep all decoded reports in a
// small FIFO and let getSensorEvent() drain them across control ticks.
// SH2 accepts a 384-byte incoming payload. The four reports used by this
// project are 10-14 bytes apiece, so 48 slots retain every report from a
// maximally packed payload after a delayed poll.
static constexpr uint8_t SENSOR_EVENT_QUEUE_LEN = 48;
static sh2_SensorValue_t _sensor_event_queue[SENSOR_EVENT_QUEUE_LEN];
static uint8_t _sensor_event_head = 0;
static uint8_t _sensor_event_tail = 0;
static uint8_t _sensor_event_count = 0;
static uint32_t _sensor_event_overflows = 0;
static uint32_t _sensor_decode_errors = 0;
static uint32_t _transport_errors = 0;
// SPI reads use one header transaction followed by one full-payload
// transaction. The BNO086 advances the SHTP channel sequence for both, so
// successfully delivered payloads normally advance by two.
static constexpr uint8_t SHTP_CHANNEL_COUNT = 8;
static constexpr uint8_t GYRO_RV_SHTP_CHANNEL = 5;
static bool _spi_rx_seq_valid[SHTP_CHANNEL_COUNT] = {};
static uint8_t _spi_rx_next_seq[SHTP_CHANNEL_COUNT] = {};
static uint32_t _spi_rx_sequence_gaps[SHTP_CHANNEL_COUNT] = {};
static bool _reset_occurred = false;

static int i2chal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len);
static int i2chal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                       uint32_t *t_us);
static void i2chal_close(sh2_Hal_t *self);
static int i2chal_open(sh2_Hal_t *self);

static int uarthal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len);
static int uarthal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                        uint32_t *t_us);
static void uarthal_close(sh2_Hal_t *self);
static int uarthal_open(sh2_Hal_t *self);

static bool spihal_wait_for_int(uint32_t timeout_us);
static int spihal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len);
static int spihal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                       uint32_t *t_us);
static void spihal_close(sh2_Hal_t *self);
static int spihal_open(sh2_Hal_t *self);

static uint32_t hal_getTimeUs(sh2_Hal_t *self);
static void hal_callback(void *cookie, sh2_AsyncEvent_t *pEvent);
static void sensorHandler(void *cookie, sh2_SensorEvent_t *pEvent);
static void hal_hardwareReset(void);

/**
 * @brief Construct a new Adafruit_BNO08x::Adafruit_BNO08x object
 *
 */

/**
 * @brief Construct a new Adafruit_BNO08x::Adafruit_BNO08x object
 *
 * @param reset_pin The arduino pin # connected to the BNO Reset pin
 */
Adafruit_BNO08x::Adafruit_BNO08x(int8_t reset_pin) { _reset_pin = reset_pin; }

/**
 * @brief Destroy the Adafruit_BNO08x::Adafruit_BNO08x object
 *
 */
Adafruit_BNO08x::~Adafruit_BNO08x(void) {
  // if (temp_sensor)
  //   delete temp_sensor;
}

/*!
 *    @brief  Sets up the hardware and initializes I2C
 *    @param  i2c_address
 *            The I2C address to be used.
 *    @param  wire
 *            The Wire object to be used for I2C connections.
 *    @param  sensor_id
 *            The unique ID to differentiate the sensors from others
 *    @return True if initialization was successful, otherwise false.
 */
bool Adafruit_BNO08x::begin_I2C(uint8_t i2c_address, TwoWire *wire,
                                int32_t sensor_id) {
  if (i2c_dev) {
    delete i2c_dev; // remove old interface
  }

  i2c_dev = new Adafruit_I2CDevice(i2c_address, wire);

  if (!i2c_dev->begin()) {
    Serial.println(F("I2C address not found"));
    return false;
  }

  _HAL.open = i2chal_open;
  _HAL.close = i2chal_close;
  _HAL.read = i2chal_read;
  _HAL.write = i2chal_write;
  _HAL.getTimeUs = hal_getTimeUs;

  return _init(sensor_id);
}

/**
 *  @brief  Sets up the hardware and initializes UART
 *
 * @param serial Pointer to Stream (HardwareSerial/SoftwareSerial) interface
 * @param sensor_id
 *            The user-defined ID to differentiate different sensors
 * @return  true if initialization was successful, otherwise false.
 */
bool Adafruit_BNO08x::begin_UART(HardwareSerial *serial, int32_t sensor_id) {
  uart_dev = serial;

  _HAL.open = uarthal_open;
  _HAL.close = uarthal_close;
  _HAL.read = uarthal_read;
  _HAL.write = uarthal_write;
  _HAL.getTimeUs = hal_getTimeUs;

  return _init(sensor_id);
}

/*!
 *    @brief  Sets up the hardware and initializes hardware SPI
 *    @param  cs_pin The arduino pin # connected to chip select
 *    @param  int_pin The arduino pin # connected to BNO08x INT
 *    @param  theSPI The SPI object to be used for SPI connections.
 *    @param  sensor_id
 *            The user-defined ID to differentiate different sensors
 *    @return true if initialization was successful, otherwise false.
 */
bool Adafruit_BNO08x::begin_SPI(uint8_t cs_pin, uint8_t int_pin,
                                SPIClass *theSPI, int32_t sensor_id) {
  i2c_dev = NULL;

  _int_pin = int_pin;
  pinMode(_int_pin, INPUT_PULLUP);

  if (spi_dev) {
    delete spi_dev; // remove old interface
  }
  spi_dev = new Adafruit_SPIDevice(cs_pin,
                                   3000000,               // frequency — BNO086 datasheet max;
                                                           // see components/characterization/IMU_Adafruit
                                   SPI_BITORDER_MSBFIRST, // bit order
                                   SPI_MODE3,             // data mode
                                   theSPI);
  if (!spi_dev->begin()) {
    return false;
  }

  _HAL.open = spihal_open;
  _HAL.close = spihal_close;
  _HAL.read = spihal_read;
  _HAL.write = spihal_write;
  _HAL.getTimeUs = hal_getTimeUs;

  return _init(sensor_id);
}

/*!  @brief Initializer for post i2c/spi init
 *   @param sensor_id Optional unique ID for the sensor set
 *   @returns True if chip identified and initialized
 */
bool Adafruit_BNO08x::_init(int32_t sensor_id) {
  int status;

  clearSensorEvents();
  _reset_occurred = false;

  hardwareReset();

  // Open SH2 interface (also registers non-sensor event handler.)
  status = sh2_open(&_HAL, hal_callback, NULL);
  if (status != SH2_OK) {
    return false;
  }

  // This board has PS0 and PS1 bridged high for SPI selection, so the BNO086
  // WAKE input is not available to the Teensy. Once boot advertisements are
  // drained the hub can sleep before the first host command. The old HAL made
  // that command appear to work by silently resetting the chip after a 500 ms
  // timeout. Make the reset-as-wake handshake explicit and bounded instead:
  // service exactly until SH2 observes RESET, then transmit while the hub is
  // awake. Continuous enabled reports keep it awake after initialization.
  hardwareReset();
  const uint32_t wake_start_us = micros();
  while (digitalRead(_int_pin) != LOW &&
         (uint32_t)(micros() - wake_start_us) < 500000UL) {}
  if (digitalRead(_int_pin) != LOW) {
    _transport_errors++;
    sh2_close();
    return false;
  }
  // Match the successful transaction ordering from the characterization
  // driver: service one boot transfer, then issue the host command before the
  // no-WAKE wiring lets the hub return to sleep.
  sh2_service();
  _reset_occurred = false;

  // Check connection partially by getting the product id's
  memset(&prodIds, 0, sizeof(prodIds));
  status = sh2_getProdIds(&prodIds);
  if (status != SH2_OK) {
    // sh2_open() owns the library's sole SHTP instance. Release it on every
    // post-open failure so a later begin_SPI() is a real retry.
    sh2_close();
    return false;
  }

  // Register sensor listener
  sh2_setSensorCallback(sensorHandler, NULL);

  return true;
}

/**
 * @brief Reset the device using the Reset pin
 *
 */
void Adafruit_BNO08x::hardwareReset(void) {
  clearSensorEvents();
  _reset_occurred = false;
  hal_hardwareReset();
}

/**
 * @brief Check if a reset has occured
 *
 * @return true: a reset has occured false: no reset has occoured
 */
bool Adafruit_BNO08x::wasReset(void) {
  bool x = _reset_occurred;
  _reset_occurred = false;

  return x;
}

/**
 * @brief Fill the given sensor value object with a new report
 *
 * @param value Pointer to an sh2_SensorValue_t struct to fil
 * @return true: The report object was filled with a new report
 * @return false: No new report available to fill
 */
bool Adafruit_BNO08x::getSensorEvent(sh2_SensorValue_t *value) {
  // A poll with no ready interrupt must be a fast false result. Upstream calls
  // sh2_service() unconditionally, and its SPI HAL can otherwise turn "no
  // sample yet" into a long control-loop stall.
  if (_sensor_event_count == 0) {
    if (spi_dev != NULL && digitalRead(_int_pin) != LOW) return false;
    sh2_service();
  }
  if (_sensor_event_count == 0) return false;

  *value = _sensor_event_queue[_sensor_event_tail];
  _sensor_event_tail = (uint8_t)((_sensor_event_tail + 1) % SENSOR_EVENT_QUEUE_LEN);
  _sensor_event_count--;
  return true;
}

bool Adafruit_BNO08x::hasQueuedSensorEvent(void) {
  return _sensor_event_count > 0;
}

void Adafruit_BNO08x::clearSensorEvents(void) {
  _sensor_event_head = 0;
  _sensor_event_tail = 0;
  _sensor_event_count = 0;
}

uint32_t Adafruit_BNO08x::sensorEventOverflowCount(void) {
  return _sensor_event_overflows;
}

uint32_t Adafruit_BNO08x::sensorDecodeErrorCount(void) {
  return _sensor_decode_errors;
}

uint32_t Adafruit_BNO08x::transportErrorCount(void) {
  return _transport_errors;
}

uint32_t Adafruit_BNO08x::gyroRvSequenceGapCount(void) {
  return _spi_rx_sequence_gaps[GYRO_RV_SHTP_CHANNEL];
}

void Adafruit_BNO08x::resetDiagnostics(void) {
  _sensor_event_overflows = 0;
  _sensor_decode_errors = 0;
  _transport_errors = 0;
  memset(_spi_rx_seq_valid, 0, sizeof(_spi_rx_seq_valid));
  memset(_spi_rx_next_seq, 0, sizeof(_spi_rx_next_seq));
  memset(_spi_rx_sequence_gaps, 0, sizeof(_spi_rx_sequence_gaps));
}

/**
 * @brief Enable the given report type
 *
 * @param sensorId The report ID to enable
 * @param interval_us The update interval for reports to be generated, in
 * microseconds
 * @return true: success false: failure
 */
bool Adafruit_BNO08x::enableReport(sh2_SensorId_t sensorId,
                                   uint32_t interval_us, bool always_on) {
  static sh2_SensorConfig_t config;

  // These sensor options are disabled or not used in most cases
  config.changeSensitivityEnabled = false;
  config.wakeupEnabled = false;
  config.changeSensitivityRelative = false;
  config.alwaysOnEnabled = always_on;
  config.changeSensitivity = 0;
  config.batchInterval_us = 0;
  config.sensorSpecific = 0;

  config.reportInterval_us = interval_us;
  int status = sh2_setSensorConfig(sensorId, &config);

  if (status != SH2_OK) {
    return false;
  }

  return true;
}

/**************************************** I2C interface
 * ***********************************************************/

static int i2chal_open(sh2_Hal_t *self) {
  // Serial.println("I2C HAL open");
  uint8_t softreset_pkt[] = {5, 0, 1, 0, 1};
  bool success = false;
  for (uint8_t attempts = 0; attempts < 5; attempts++) {
    if (i2c_dev->write(softreset_pkt, 5)) {
      success = true;
      break;
    }
    delay(30);
  }
  if (!success)
    return -1;
  delay(300);
  return 0;
}

static void i2chal_close(sh2_Hal_t *self) {
  // Serial.println("I2C HAL close");
}

static int i2chal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                       uint32_t *t_us) {
  // Serial.println("I2C HAL read");

  // uint8_t *pBufferOrig = pBuffer;

  uint8_t header[4];
  if (!i2c_dev->read(header, 4)) {
    return 0;
  }

  // Determine amount to read
  uint16_t packet_size = (uint16_t)header[0] | (uint16_t)header[1] << 8;
  // Unset the "continue" bit
  packet_size &= ~0x8000;

  /*
  Serial.print("Read SHTP header. ");
  Serial.print("Packet size: ");
  Serial.print(packet_size);
  Serial.print(" & buffer size: ");
  Serial.println(len);
  */

  size_t i2c_buffer_max = i2c_dev->maxBufferSize();

  if (packet_size > len) {
    // packet wouldn't fit in our buffer
    return 0;
  }
  // the number of non-header bytes to read
  uint16_t cargo_remaining = packet_size;
  uint8_t i2c_buffer[i2c_buffer_max];
  uint16_t read_size;
  uint16_t cargo_read_amount = 0;
  bool first_read = true;

  while (cargo_remaining > 0) {
    if (first_read) {
      read_size = min(i2c_buffer_max, (size_t)cargo_remaining);
    } else {
      read_size = min(i2c_buffer_max, (size_t)cargo_remaining + 4);
    }

    // Serial.print("Reading from I2C: "); Serial.println(read_size);
    // Serial.print("Remaining to read: "); Serial.println(cargo_remaining);

    if (!i2c_dev->read(i2c_buffer, read_size)) {
      return 0;
    }

    if (first_read) {
      // The first time we're saving the "original" header, so include it in the
      // cargo count
      cargo_read_amount = read_size;
      memcpy(pBuffer, i2c_buffer, cargo_read_amount);
      first_read = false;
    } else {
      // this is not the first read, so copy from 4 bytes after the beginning of
      // the i2c buffer to skip the header included with every new i2c read and
      // don't include the header in the amount of cargo read
      cargo_read_amount = read_size - 4;
      memcpy(pBuffer, i2c_buffer + 4, cargo_read_amount);
    }
    // advance our pointer by the amount of cargo read
    pBuffer += cargo_read_amount;
    // mark the cargo as received
    cargo_remaining -= cargo_read_amount;
  }

  /*
  for (int i=0; i<packet_size; i++) {
    Serial.print(pBufferOrig[i], HEX);
    Serial.print(", ");
    if (i % 16 == 15) Serial.println();
  }
  Serial.println();
  */

  return packet_size;
}

static int i2chal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len) {
  size_t i2c_buffer_max = i2c_dev->maxBufferSize();

  /*
  Serial.print("I2C HAL write packet size: ");
  Serial.print(len);
  Serial.print(" & max buffer size: ");
  Serial.println(i2c_buffer_max);
  */

  uint16_t write_size = min(i2c_buffer_max, len);
  if (!i2c_dev->write(pBuffer, write_size)) {
    return 0;
  }

  return write_size;
}

/**************************************** UART interface
 * ***********************************************************/

static int uarthal_open(sh2_Hal_t *self) {
  // Serial.println("UART HAL open");
  uart_dev->begin(3000000);

  // flush input
  while (uart_dev->available()) {
    uart_dev->read();
    yield();
  }

  // send a software reset
  uint8_t softreset_pkt[] = {0x7E, 1, 5, 0, 1, 0, 1, 0x7E};
  for (int i = 0; i < sizeof(softreset_pkt); i++) {
    uart_dev->write(softreset_pkt[i]);
    delay(1);
  }

  return 0;
}

static void uarthal_close(sh2_Hal_t *self) {
  // Serial.println("UART HAL close");
  uart_dev->end();
}

static int uarthal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                        uint32_t *t_us) {
  uint8_t c;
  uint16_t packet_size = 0;

  // Serial.println("UART HAL read");

  // read packet start
  while (1) {
    yield();

    if (!uart_dev->available()) {
      continue;
    }
    c = uart_dev->read();
    // Serial.print(c, HEX); Serial.print(", ");
    if (c == 0x7E) {
      break;
    }
  }

  // read protocol id
  while (uart_dev->available() < 2) {
    yield();
  }
  c = uart_dev->read();
  // Serial.print(c, HEX); Serial.print(", ");
  if (c == 0x7E) {
    c = uart_dev->read();
    // Serial.print(c, HEX); Serial.print(", ");
    if (c != 0x01) {
      return 0;
    }
  } else if (c != 0x01) {
    return 0;
  }

  while (true) {
    yield();

    if (!uart_dev->available()) {
      continue;
    }
    c = uart_dev->read();
    // Serial.print(c, HEX); Serial.print(", ");
    if (c == 0x7E) {
      break;
    }
    if (c == 0x7D) {
      // escape!
      while (!uart_dev->available()) {
        continue;
      }
      c = uart_dev->read();
      c ^= 0x20;
    }
    pBuffer[packet_size] = c;
    packet_size++;
  }

  /*
  Serial.print("Read UART packet size: ");
  Serial.println(packet_size);
  for (int i=0; i<packet_size; i++) {
    Serial.print(pBuffer[i], HEX);
    Serial.print(", ");
    if (i % 16 == 15) Serial.println();
  }
  Serial.println();
  */

  return packet_size;
}

static int uarthal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len) {
  uint8_t c;

  // Serial.print("UART HAL write packet size: ");
  // Serial.println(len);

  // start byte
  uart_dev->write(0x7E);
  delay(1);
  // protocol id
  uart_dev->write(0x01);
  delay(1);

  for (int i = 0; i < len; i++) {
    c = pBuffer[i];
    if ((c == 0x7E) || (c == 0x7D)) {
      uart_dev->write(0x7D); // control
      delay(1);
      c ^= 0x20;
    }
    uart_dev->write(c);
    delay(1);
  }
  // end byte
  uart_dev->write(0x7E);

  return len;
}

/**************************************** UART interface
 * ***********************************************************/

static int spihal_open(sh2_Hal_t *self) {
  // Serial.println("SPI HAL open");

  // Startup is the one legitimate long wait: the BNO08x needs about 90 ms of
  // internal initialization after reset. Runtime reads never use this budget.
  if (spihal_wait_for_int(500000UL)) return 0;
  _transport_errors++;
  return -1;
}

static bool spihal_wait_for_int(uint32_t timeout_us) {
  // Tight spin instead of delay(1)-per-poll — eliminates up to ~2 ms dead
  // time per SHTP read (header + payload each wait for INT); see
  // components/characterization/IMU_Adafruit/IMU_adafruit.MD.
  uint32_t start = micros();
  while ((micros() - start) < timeout_us) {
    if (!digitalRead(_int_pin))
      return true;
  }
  return false;
}

static void spihal_close(sh2_Hal_t *self) {
  // Serial.println("SPI HAL close");
}

static int spihal_read(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len,
                       uint32_t *t_us) {
  // Serial.println("SPI HAL read");

  uint16_t packet_size = 0;

  // H_INTN high means there is no packet ready. Synchronous SH2 operations
  // call this repeatedly, so this path must be immediate and non-destructive.
  if (digitalRead(_int_pin) != LOW) return 0;
  if (t_us) *t_us = micros();

  if (!spi_dev->read(pBuffer, 4, 0x00)) {
    _transport_errors++;
    return 0;
  }

  // Determine amount to read
  packet_size = (uint16_t)pBuffer[0] | (uint16_t)pBuffer[1] << 8;
  // Unset the "continue" bit
  packet_size &= ~0x8000;

  /*
  Serial.print("Read SHTP header. ");
  Serial.print("Packet size: ");
  Serial.print(packet_size);
  Serial.print(" & buffer size: ");
  Serial.println(len);
  */

  if (packet_size < 4 || packet_size > len) {
    _transport_errors++;
    return 0;
  }

  // The payload-ready handshake is normally measured in microseconds. Bound
  // a failed transfer so it cannot wedge the real-time loop.
  if (!spihal_wait_for_int(5000UL)) {
    _transport_errors++;
    return 0;
  }

  if (!spi_dev->read(pBuffer, packet_size, 0x00)) {
    _transport_errors++;
    return 0;
  }

  const uint8_t channel = pBuffer[2];
  const uint8_t sequence = pBuffer[3];
  if (channel < SHTP_CHANNEL_COUNT) {
    if (_spi_rx_seq_valid[channel] && sequence != _spi_rx_next_seq[channel]) {
      const uint8_t gap = (uint8_t)(sequence - _spi_rx_next_seq[channel]);
      if (gap < 64) _spi_rx_sequence_gaps[channel] += gap;
    }
    _spi_rx_seq_valid[channel] = true;
    _spi_rx_next_seq[channel] = sequence + 2;
  }

  return packet_size;
}

static int spihal_write(sh2_Hal_t *self, uint8_t *pBuffer, unsigned len) {
  // Serial.print("SPI HAL write packet size: ");
  // Serial.println(len);

  // Writes occur during initialization/report setup. Never hide a hardware
  // reset down in the HAL; return failure so the driver can recover coherently.
  // A healthy runtime write normally starts immediately, but boot-time
  // product-ID traffic can leave the hub busy for tens of milliseconds. Keep
  // a finite 100 ms ceiling: long enough for the real device, far below the
  // former unbounded SHTP retry.
  if (!spihal_wait_for_int(100000UL)) {
    _transport_errors++;
    // SHTP treats zero as "busy, retry indefinitely". A bounded timeout is a
    // real transport failure and must be negative so txProcess() can unwind.
    return -1;
  }

  if (!spi_dev->write(pBuffer, len)) {
    _transport_errors++;
    return -1;
  }

  return len;
}

/**************************************** HAL interface
 * ***********************************************************/

static void hal_hardwareReset(void) {
  if (_reset_pin != -1) {
    // Serial.println("BNO08x Hardware reset");

    pinMode(_reset_pin, OUTPUT);
    digitalWrite(_reset_pin, HIGH);
    delay(10);
    digitalWrite(_reset_pin, LOW);
    delay(10);
    digitalWrite(_reset_pin, HIGH);
    delay(10);
  }
}

static uint32_t hal_getTimeUs(sh2_Hal_t *self) {
  uint32_t t = micros();
  // Serial.printf("I2C HAL get time: %d\n", t);
  return t;
}

static void hal_callback(void *cookie, sh2_AsyncEvent_t *pEvent) {
  // If we see a reset, set a flag so that sensors will be reconfigured.
  if (pEvent->eventId == SH2_RESET) {
    // Serial.println("Reset!");
    _reset_occurred = true;
    memset(_spi_rx_seq_valid, 0, sizeof(_spi_rx_seq_valid));
  }
}

// Handle sensor events.
static void sensorHandler(void *cookie, sh2_SensorEvent_t *event) {
  int rc;

  // Serial.println("Got an event!");

  sh2_SensorValue_t value;
  rc = sh2_decodeSensorEvent(&value, event);
  if (rc != SH2_OK) {
    _sensor_decode_errors++;
    return;
  }

  // Preserve the newest evidence if an unexpectedly large packet overflows
  // the FIFO. Normal packets are far smaller than 48 reports, and IMU.cpp
  // drains up to sixteen reports per 500 Hz tick.
  if (_sensor_event_count == SENSOR_EVENT_QUEUE_LEN) {
    _sensor_event_tail = (uint8_t)((_sensor_event_tail + 1) % SENSOR_EVENT_QUEUE_LEN);
    _sensor_event_count--;
    _sensor_event_overflows++;
  }
  _sensor_event_queue[_sensor_event_head] = value;
  _sensor_event_head = (uint8_t)((_sensor_event_head + 1) % SENSOR_EVENT_QUEUE_LEN);
  _sensor_event_count++;
}
