# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:01:26 2024

@author: dariu
"""


import asyncio
from bleak import BleakScanner

async def scan():
    devices = await BleakScanner.discover()
    for device in devices:
        print(device)

# Wywołaj funkcję asynchroniczną za pomocą await
await scan()

import asyncio
from bleak import BleakClient

async def read_ble_device(mac_address):
    async with BleakClient(mac_address) as client:
        # Pobierz listę wszystkich charakterystyk
        services = await client.get_services()
        for service in services:
            print(f"Service UUID: {service.uuid}")
            for char in service.characteristics:
                # Odczytaj wartość charakterystyki
                value = await client.read_gatt_char(char.uuid)
                print(f"Characteristic UUID: {char.uuid}, Value: {value}")

# Wybierz adres MAC urządzenia, z którym chcesz się połączyć
device_mac_address = "DC:D4:66:B3:04:98"

# Uruchom funkcję odczytu dla wybranego urządzenia bezpośrednio w komórce
await read_ble_device(device_mac_address)

import asyncio
from bleak import BleakClient

async def read_ble_device(mac_address):
    async with BleakClient(mac_address) as client:
        # Pobierz listę wszystkich usług
        services = await client.get_services()
        for service in services:
            print(f"Service UUID: {service.uuid}")
            # Dla każdej usługi pobierz listę charakterystyk
            characteristics = service.characteristics
            for char in characteristics:
                # Odczytaj wartość charakterystyki
                value = await client.read_gatt_char(char.uuid)
                print(f"Characteristic UUID: {char.uuid}, Value: {value}")

# Wybierz adres MAC urządzenia, z którym chcesz się połączyć
device_mac_address = "DC:D4:66:B3:04:98"

# Uruchom funkcję odczytu dla wybranego urządzenia
asyncio.run(read_ble_device(device_mac_address))