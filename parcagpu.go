package parcagpu

import (
	"io/fs"
	"path/filepath"
)

type platformType uint8

const (
	CupTI platformType = iota
)

func Open(platform platformType) (fs.File, error) {
	switch platform {
	case CupTI:
		return archive.Open(filepath.Join(archivePrefix, "libcolacupti.so"))
	default:
		return nil, fs.ErrNotExist
	}
}
