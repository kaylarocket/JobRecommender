import 'package:flutter/material.dart';

class FormInput extends StatelessWidget {
  const FormInput({super.key, required this.label, required this.controller, this.hint, this.obscure = false, this.keyboardType = TextInputType.text});

  final String label;
  final TextEditingController controller;
  final String? hint;
  final bool obscure;
  final TextInputType keyboardType;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Padding(
          padding: const EdgeInsets.only(left: 4, bottom: 6),
          child: Text(label, style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
        ),
        TextField(
          controller: controller,
          obscureText: obscure,
          keyboardType: keyboardType,
          decoration: InputDecoration(
            hintText: hint,
          ),
        ),
      ],
    );
  }
}
