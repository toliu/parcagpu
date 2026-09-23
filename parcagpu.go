package parcagpu

import (
	"io/fs"
	"path/filepath"
)

type platformType uint8

const (
	CUPTI platformType = iota
	MSPTI
)

func Open(platform platformType) (fs.File, error) {
	var filename string
	switch platform {
	case CUPTI:
		filename = `libcolacupti.so`
	case MSPTI:
		filename = `libcolamspti.so`
	default:
		return nil, fs.ErrNotExist
	}
	return archive.Open(filepath.Join(archivePrefix, filename))
}
